/**
 * Orchestration Language Model Strategy - Implementation using `@sap-ai-sdk/orchestration`.
 *
 * This strategy is stateless - it holds only a reference to the OrchestrationClient class.
 * All tenant-specific configuration flows through method parameters for security.
 */
import type {
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type {
  ChatCompletionTool,
  ChatMessage,
  LlmModelParams,
  OrchestrationClient,
  OrchestrationModuleConfig,
} from "@sap-ai-sdk/orchestration";

import { parseProviderOptions } from "@ai-sdk/provider-utils";

import type {
  OrchestrationModelSettings,
  PromptTemplateRef,
  PromptTemplateRefByID,
  SAPAIModelSettings,
} from "./sap-ai-settings.js";
import type { LanguageModelAPIStrategy, LanguageModelStrategyConfig } from "./sap-ai-strategy.js";

import { convertToSAPMessages } from "./convert-to-sap-messages.js";
import { convertToAISDKError, normalizeHeaders } from "./sap-ai-error.js";
import { getProviderName, sapAILanguageModelProviderOptions } from "./sap-ai-provider-options.js";
import {
  buildGenerateResult,
  buildModelParams,
  convertResponseFormat,
  convertToolsToSAPFormat,
  createAISDKRequestBodySummary,
  createStreamTransformer,
  mapToolChoice,
  type ParamMapping,
  type SAPToolChoice,
  type SDKResponse,
  type SDKStreamChunk,
  StreamIdGenerator,
} from "./strategy-utils.js";
import { VERSION } from "./version.js";

/**
 * Extended prompt templating structure with tools and response_format.
 * The SAP SDK type doesn't expose these properties, but they are set when building the config.
 * @internal
 */
interface ExtendedPromptTemplating {
  prompt: {
    response_format?: unknown;
    template?: unknown[];
    template_ref?: unknown;
    tools?: unknown;
  };
}

/**
 * Extended model parameters for SAP Orchestration API.
 * Includes standard LLM parameters plus SAP-specific extensions.
 * @internal
 */
type SAPModelParams = LlmModelParams & {
  parallel_tool_calls?: boolean;
  seed?: number;
  stop?: string[];
  top_k?: number;
};

/**
 * Type guard to check if a PromptTemplateRef is by ID.
 * @param ref - The template reference to check.
 * @returns True if the reference is by ID.
 * @internal
 */
function isTemplateRefById(ref: PromptTemplateRef): ref is PromptTemplateRefByID {
  return "id" in ref;
}

/**
 * Parameter mappings for override resolution and camelCase conversion.
 * @internal
 */
const PARAM_MAPPINGS: readonly ParamMapping[] = [
  { camelCaseKey: "maxTokens", optionKey: "maxOutputTokens", outputKey: "max_tokens" },
  { camelCaseKey: "temperature", optionKey: "temperature", outputKey: "temperature" },
  { camelCaseKey: "topP", optionKey: "topP", outputKey: "top_p" },
  { camelCaseKey: "topK", optionKey: "topK", outputKey: "top_k" },
  {
    camelCaseKey: "frequencyPenalty",
    optionKey: "frequencyPenalty",
    outputKey: "frequency_penalty",
  },
  { camelCaseKey: "presencePenalty", optionKey: "presencePenalty", outputKey: "presence_penalty" },
  { camelCaseKey: "seed", optionKey: "seed", outputKey: "seed" },
  { camelCaseKey: "parallel_tool_calls", outputKey: "parallel_tool_calls" },
] as const;

/**
 * Type for the OrchestrationClient class constructor.
 * @internal
 */
type OrchestrationClientClass = typeof OrchestrationClient;

/**
 * Orchestration Language Model Strategy.
 *
 * Implements language model operations using the SAP AI SDK Orchestration API.
 * This class is stateless - it only holds a reference to the OrchestrationClient class.
 * @internal
 */
export class OrchestrationLanguageModelStrategy implements LanguageModelAPIStrategy {
  private readonly ClientClass: OrchestrationClientClass;

  /**
   * Creates a new OrchestrationLanguageModelStrategy.
   * @param ClientClass - The OrchestrationClient class from `@sap-ai-sdk/orchestration`.
   */
  constructor(ClientClass: OrchestrationClientClass) {
    this.ClientClass = ClientClass;
  }

  /**
   * Generates a single completion (non-streaming).
   *
   * Builds orchestration configuration, converts messages, validates parameters,
   * calls SAP AI SDK, and processes the response.
   * @param config - The language model strategy configuration.
   * @param settings - The SAP AI model settings.
   * @param options - The Vercel AI SDK language model call options.
   * @returns A Promise resolving to the generation result.
   * @throws {Error} When the SAP AI SDK request fails.
   */
  async doGenerate(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    try {
      const { messages, orchestrationConfig, placeholderValues, toolChoice, warnings } =
        await this.buildOrchestrationConfig(config, settings, options);

      const client = this.createClient(config, orchestrationConfig);

      const requestBody = this.buildRequestBody(
        messages,
        orchestrationConfig,
        placeholderValues,
        toolChoice,
      );

      const response = await client.chatCompletion(
        requestBody,
        options.abortSignal ? { signal: options.abortSignal } : undefined,
      );

      return buildGenerateResult({
        modelId: config.modelId,
        providerName: getProviderName(config.provider),
        requestBody,
        response: response as SDKResponse,
        responseHeaders: normalizeHeaders(response.rawResponse.headers),
        version: VERSION,
        warnings,
      });
    } catch (error) {
      throw convertToAISDKError(error, {
        operation: "doGenerate",
        requestBody: createAISDKRequestBodySummary(options),
        url: "sap-ai:orchestration",
      });
    }
  }

  /**
   * Generates a streaming completion.
   *
   * Builds orchestration configuration, creates streaming client, and transforms
   * the stream with proper event handling (text blocks, tool calls, finish reason).
   * @param config - The language model strategy configuration.
   * @param settings - The SAP AI model settings.
   * @param options - The Vercel AI SDK language model call options.
   * @returns A Promise resolving to the streaming result.
   * @throws {Error} When the SAP AI SDK streaming request fails.
   */
  async doStream(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    try {
      const { messages, orchestrationConfig, placeholderValues, toolChoice, warnings } =
        await this.buildOrchestrationConfig(config, settings, options);

      const client = this.createClient(config, orchestrationConfig);

      const requestBody = this.buildRequestBody(
        messages,
        orchestrationConfig,
        placeholderValues,
        toolChoice,
      );

      const streamResponse = await client.stream(requestBody, options.abortSignal, {
        promptTemplating: { include_usage: true },
      });

      const idGenerator = new StreamIdGenerator();
      const responseId = idGenerator.generateResponseId();

      const transformedStream = createStreamTransformer({
        convertToAISDKError,
        idGenerator,
        includeRawChunks: options.includeRawChunks ?? false,
        modelId: config.modelId,
        options,
        providerName: getProviderName(config.provider),
        responseId,
        sdkStream: streamResponse.stream as AsyncIterable<SDKStreamChunk>,
        streamResponseGetFinishReason: () => streamResponse.getFinishReason(),
        streamResponseGetTokenUsage: () => streamResponse.getTokenUsage(),
        url: "sap-ai:orchestration",
        version: VERSION,
        warnings,
      });

      return {
        request: {
          body: requestBody as unknown,
        },
        stream: transformedStream,
      };
    } catch (error) {
      throw convertToAISDKError(error, {
        operation: "doStream",
        requestBody: createAISDKRequestBodySummary(options),
        url: "sap-ai:orchestration",
      });
    }
  }

  /**
   * Builds the SAP AI SDK orchestration configuration from Vercel AI SDK call options.
   * @param config - The language model strategy configuration.
   * @param settings - The SAP AI model settings.
   * @param options - The Vercel AI SDK language model call options.
   * @returns A Promise resolving to the messages, orchestration config, and warnings.
   * @internal
   */
  private async buildOrchestrationConfig(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<{
    messages: ChatMessage[];
    orchestrationConfig: OrchestrationModuleConfig;
    placeholderValues?: Record<string, string>;
    toolChoice?: SAPToolChoice;
    warnings: SharedV3Warning[];
  }> {
    const providerName = getProviderName(config.provider);
    const sapOptions = await parseProviderOptions({
      provider: providerName,
      providerOptions: options.providerOptions,
      schema: sapAILanguageModelProviderOptions,
    });

    const warnings: SharedV3Warning[] = [];

    const orchSettings = settings as OrchestrationModelSettings;

    const messages = convertToSAPMessages(options.prompt, {
      escapeTemplatePlaceholders:
        sapOptions?.escapeTemplatePlaceholders ?? orchSettings.escapeTemplatePlaceholders ?? true,
      includeReasoning: sapOptions?.includeReasoning ?? orchSettings.includeReasoning ?? false,
    });

    // Handle tools: settings.tools take precedence if options.tools is empty
    let tools: ChatCompletionTool[] | undefined;
    const settingsTools = orchSettings.tools;
    const optionsTools = options.tools;

    if (settingsTools && settingsTools.length > 0 && optionsTools && optionsTools.length > 0) {
      warnings.push({
        message:
          "Both settings.tools and call options.tools were provided; preferring call options.tools.",
        type: "other",
      });
    }

    if (settingsTools && settingsTools.length > 0 && (!optionsTools || optionsTools.length === 0)) {
      // Use pre-configured tools from settings
      tools = settingsTools;
    } else if (optionsTools && optionsTools.length > 0) {
      // Convert AI SDK tools to SAP format using shared helper
      const toolsResult = convertToolsToSAPFormat<ChatCompletionTool>(optionsTools);
      tools = toolsResult.tools;
      warnings.push(...toolsResult.warnings);
    }

    // Build model parameters using shared helper
    const { modelParams: baseModelParams, warnings: paramWarnings } = buildModelParams({
      options,
      paramMappings: PARAM_MAPPINGS,
      providerModelParams: sapOptions?.modelParams as Record<string, unknown> | undefined,
      settingsModelParams: orchSettings.modelParams as Record<string, unknown> | undefined,
    });
    const modelParams = baseModelParams as SAPModelParams;
    warnings.push(...paramWarnings);

    // Map Vercel AI SDK toolChoice to SAP Orchestration tool_choice
    const toolChoice = mapToolChoice(options.toolChoice);

    // Convert response format using shared helper
    const { responseFormat, warning: responseFormatWarning } = convertResponseFormat(
      options.responseFormat,
      orchSettings.responseFormat,
    );
    if (responseFormatWarning) {
      warnings.push(responseFormatWarning);
    }

    // Determine prompt template reference: providerOptions override settings
    const promptTemplateRef = sapOptions?.promptTemplateRef ?? orchSettings.promptTemplateRef;

    // Build the prompt configuration - either using template_ref or inline template
    // Note: We use type assertion because the SDK's Xor type doesn't allow
    // additional properties like tools/response_format alongside template_ref
    const promptConfig: Record<string, unknown> = promptTemplateRef
      ? {
          template_ref: isTemplateRefById(promptTemplateRef)
            ? {
                id: promptTemplateRef.id,
                ...(promptTemplateRef.scope && { scope: promptTemplateRef.scope }),
              }
            : {
                name: promptTemplateRef.name,
                scenario: promptTemplateRef.scenario,
                version: promptTemplateRef.version,
                ...(promptTemplateRef.scope && { scope: promptTemplateRef.scope }),
              },
          ...(tools && tools.length > 0 ? { tools } : {}),
          ...(responseFormat ? { response_format: responseFormat } : {}),
        }
      : {
          template: [],
          ...(tools && tools.length > 0 ? { tools } : {}),
          ...(responseFormat ? { response_format: responseFormat } : {}),
        };

    const orchestrationConfig: OrchestrationModuleConfig = {
      promptTemplating: {
        model: {
          name: config.modelId,
          params: modelParams,
          ...(orchSettings.modelVersion ? { version: orchSettings.modelVersion } : {}),
        },
        prompt: promptConfig as OrchestrationModuleConfig["promptTemplating"]["prompt"],
      },
      ...(orchSettings.masking && Object.keys(orchSettings.masking as object).length > 0
        ? { masking: orchSettings.masking }
        : {}),
      ...(orchSettings.filtering && Object.keys(orchSettings.filtering as object).length > 0
        ? { filtering: orchSettings.filtering }
        : {}),
      ...(orchSettings.grounding && Object.keys(orchSettings.grounding as object).length > 0
        ? { grounding: orchSettings.grounding }
        : {}),
      ...(orchSettings.translation && Object.keys(orchSettings.translation as object).length > 0
        ? { translation: orchSettings.translation }
        : {}),
    };

    // Merge placeholderValues: providerOptions override settings
    const mergedPlaceholderValues =
      orchSettings.placeholderValues || sapOptions?.placeholderValues
        ? {
            ...orchSettings.placeholderValues,
            ...sapOptions?.placeholderValues,
          }
        : undefined;
    const placeholderValues =
      mergedPlaceholderValues && Object.keys(mergedPlaceholderValues).length > 0
        ? mergedPlaceholderValues
        : undefined;

    return {
      messages,
      orchestrationConfig,
      placeholderValues,
      toolChoice,
      warnings,
    };
  }

  /**
   * Builds the request body for SAP AI SDK chat completion or streaming.
   * @param messages - The chat messages to send.
   * @param orchestrationConfig - The orchestration module configuration.
   * @param placeholderValues - Optional placeholder values for template variables.
   * @param toolChoice - Optional tool choice configuration.
   * @returns The request body object for the SAP AI SDK.
   * @internal
   */
  private buildRequestBody(
    messages: ChatMessage[],
    orchestrationConfig: OrchestrationModuleConfig,
    placeholderValues?: Record<string, string>,
    toolChoice?: SAPToolChoice,
  ): Record<string, unknown> {
    // Type assertion: SDK's OrchestrationModuleConfig type doesn't expose prompt.tools,
    // prompt.response_format, or prompt.template_ref properties, but they are set in buildOrchestrationConfig
    const promptTemplating = orchestrationConfig.promptTemplating as ExtendedPromptTemplating;

    return {
      messages,
      model: {
        ...orchestrationConfig.promptTemplating.model,
      },
      ...(placeholderValues ? { placeholderValues } : {}),
      // Include template_ref when using Prompt Registry reference
      ...(promptTemplating.prompt.template_ref
        ? { template_ref: promptTemplating.prompt.template_ref }
        : {}),
      ...(promptTemplating.prompt.tools ? { tools: promptTemplating.prompt.tools } : {}),
      ...(toolChoice ? { tool_choice: toolChoice } : {}),
      ...(promptTemplating.prompt.response_format
        ? { response_format: promptTemplating.prompt.response_format }
        : {}),
      ...(orchestrationConfig.masking && Object.keys(orchestrationConfig.masking).length > 0
        ? { masking: orchestrationConfig.masking }
        : {}),
      ...(orchestrationConfig.filtering && Object.keys(orchestrationConfig.filtering).length > 0
        ? { filtering: orchestrationConfig.filtering }
        : {}),
      ...(orchestrationConfig.grounding && Object.keys(orchestrationConfig.grounding).length > 0
        ? { grounding: orchestrationConfig.grounding }
        : {}),
      ...(orchestrationConfig.translation && Object.keys(orchestrationConfig.translation).length > 0
        ? { translation: orchestrationConfig.translation }
        : {}),
    };
  }

  /**
   * Creates an SAP AI SDK OrchestrationClient with the given configuration.
   * @param config - The language model strategy configuration.
   * @param orchConfig - The orchestration module configuration.
   * @returns A new OrchestrationClient instance.
   * @internal
   */
  private createClient(
    config: LanguageModelStrategyConfig,
    orchConfig: OrchestrationModuleConfig,
  ): InstanceType<OrchestrationClientClass> {
    return new this.ClientClass(orchConfig, config.deploymentConfig, config.destination);
  }
}
