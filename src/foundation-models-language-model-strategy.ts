/**
 * Foundation Models Language Model Strategy - Implementation using `@sap-ai-sdk/foundation-models`.
 *
 * This strategy is stateless - it holds only a reference to the AzureOpenAiChatClient class.
 * All tenant-specific configuration flows through method parameters for security.
 */
import type {
  LanguageModelV3CallOptions,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamResult,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type {
  AzureOpenAiChatClient,
  AzureOpenAiChatCompletionParameters,
  AzureOpenAiChatCompletionTool,
} from "@sap-ai-sdk/foundation-models";

import { parseProviderOptions } from "@ai-sdk/provider-utils";

import type { FoundationModelsModelSettings, SAPAIModelSettings } from "./sap-ai-settings.js";
import type { LanguageModelAPIStrategy, LanguageModelStrategyConfig } from "./sap-ai-strategy.js";

import { convertToSAPMessages } from "./convert-to-sap-messages.js";
import { deepMerge } from "./deep-merge.js";
import { convertToAISDKError, normalizeHeaders } from "./sap-ai-error.js";
import {
  getProviderName,
  sapAILanguageModelProviderOptions,
  validateModelParamsWithWarnings,
} from "./sap-ai-provider-options.js";
import {
  applyParameterOverrides,
  buildGenerateResult,
  buildModelDeployment,
  createAISDKRequestBodySummary,
  createStreamTransformer,
  extractToolParameters,
  mapToolChoice,
  type ParamMapping,
  type SDKResponse,
  type SDKStreamChunk,
  StreamIdGenerator,
} from "./strategy-utils.js";
import { VERSION } from "./version.js";

/**
 * Parameter mappings for override resolution and camelCase conversion.
 * Foundation Models API supports additional parameters like logprobs, seed, stop, user.
 * @internal
 */
const PARAM_MAPPINGS: readonly ParamMapping[] = [
  { camelCaseKey: "maxTokens", optionKey: "maxOutputTokens", outputKey: "max_tokens" },
  { camelCaseKey: "temperature", optionKey: "temperature", outputKey: "temperature" },
  { camelCaseKey: "topP", optionKey: "topP", outputKey: "top_p" },
  {
    camelCaseKey: "frequencyPenalty",
    optionKey: "frequencyPenalty",
    outputKey: "frequency_penalty",
  },
  { camelCaseKey: "presencePenalty", optionKey: "presencePenalty", outputKey: "presence_penalty" },
  { camelCaseKey: "seed", optionKey: "seed", outputKey: "seed" },
  { camelCaseKey: "parallel_tool_calls", outputKey: "parallel_tool_calls" },
  { camelCaseKey: "logprobs", outputKey: "logprobs" },
  { camelCaseKey: "topLogprobs", outputKey: "top_logprobs" },
  { camelCaseKey: "logitBias", outputKey: "logit_bias" },
  { camelCaseKey: "user", outputKey: "user" },
  { camelCaseKey: "n", outputKey: "n" },
] as const;

/**
 * Type for the AzureOpenAiChatClient class constructor.
 * @internal
 */
type AzureOpenAiChatClientClass = typeof AzureOpenAiChatClient;

/**
 * Foundation Models Language Model Strategy.
 *
 * Implements language model operations using the SAP AI SDK Foundation Models API.
 * This class is stateless - it only holds a reference to the AzureOpenAiChatClient class.
 * @internal
 */
export class FoundationModelsLanguageModelStrategy implements LanguageModelAPIStrategy {
  private readonly ClientClass: AzureOpenAiChatClientClass;

  /**
   * Creates a new FoundationModelsLanguageModelStrategy.
   * @param ClientClass - The AzureOpenAiChatClient class from `@sap-ai-sdk/foundation-models`.
   */
  constructor(ClientClass: AzureOpenAiChatClientClass) {
    this.ClientClass = ClientClass;
  }

  /**
   * Generates a single completion (non-streaming).
   *
   * Builds request parameters, converts messages, validates parameters,
   * calls SAP AI SDK, and processes the response.
   * @param config - The strategy configuration containing model and deployment info.
   * @param settings - The language model settings.
   * @param options - The call options including prompt and parameters.
   * @returns The generation result with content, usage, and metadata.
   * @throws {Error} When the SAP AI SDK request fails.
   */
  async doGenerate(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3GenerateResult> {
    try {
      const { request, warnings } = await this.buildRequest(config, settings, options);

      const client = this.createClient(config, settings.modelVersion);

      const response = await client.run(
        request,
        options.abortSignal ? { signal: options.abortSignal } : undefined,
      );

      return buildGenerateResult({
        modelId: config.modelId,
        providerName: getProviderName(config.provider),
        requestBody: request,
        response: response as SDKResponse,
        responseHeaders: normalizeHeaders(response.rawResponse.headers),
        version: VERSION,
        warnings,
      });
    } catch (error) {
      throw convertToAISDKError(error, {
        operation: "doGenerate",
        requestBody: createAISDKRequestBodySummary(options),
        url: "sap-ai:foundation-models",
      });
    }
  }

  /**
   * Generates a streaming completion.
   *
   * Builds request parameters, creates streaming client, and transforms
   * the stream with proper event handling (text blocks, tool calls, finish reason).
   * @param config - The strategy configuration containing model and deployment info.
   * @param settings - The language model settings.
   * @param options - The call options including prompt and parameters.
   * @returns The streaming result with async iterable stream and metadata.
   * @throws {Error} When the SAP AI SDK streaming request fails.
   */
  async doStream(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult> {
    try {
      const { request, warnings } = await this.buildRequest(config, settings, options);

      const client = this.createClient(config, settings.modelVersion);

      const streamResponse = await client.stream(request, options.abortSignal);

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
        url: "sap-ai:foundation-models",
        version: VERSION,
        warnings,
      });

      return {
        request: {
          body: request as unknown,
        },
        stream: transformedStream,
      };
    } catch (error) {
      throw convertToAISDKError(error, {
        operation: "doStream",
        requestBody: createAISDKRequestBodySummary(options),
        url: "sap-ai:foundation-models",
      });
    }
  }

  /**
   * Builds the Azure OpenAI chat completion request from Vercel AI SDK call options.
   * @param config - The strategy configuration containing model and deployment info.
   * @param settings - The language model settings.
   * @param options - The call options including prompt and parameters.
   * @returns The request parameters and any warnings generated.
   * @internal
   */
  private async buildRequest(
    config: LanguageModelStrategyConfig,
    settings: SAPAIModelSettings,
    options: LanguageModelV3CallOptions,
  ): Promise<{
    request: AzureOpenAiChatCompletionParameters;
    warnings: SharedV3Warning[];
  }> {
    const providerName = getProviderName(config.provider);
    const sapOptions = await parseProviderOptions({
      provider: providerName,
      providerOptions: options.providerOptions,
      schema: sapAILanguageModelProviderOptions,
    });

    const warnings: SharedV3Warning[] = [];

    const fmSettings = settings as FoundationModelsModelSettings;

    const messages = convertToSAPMessages(options.prompt, {
      escapeTemplatePlaceholders: sapOptions?.escapeTemplatePlaceholders ?? false,
      includeReasoning: sapOptions?.includeReasoning ?? fmSettings.includeReasoning ?? false,
    });

    let tools: AzureOpenAiChatCompletionTool[] | undefined;

    const optionsTools = options.tools;

    if (optionsTools && optionsTools.length > 0) {
      tools = optionsTools
        .map((tool): AzureOpenAiChatCompletionTool | null => {
          if (tool.type === "function") {
            const { parameters, warning } = extractToolParameters(tool);
            if (warning) {
              warnings.push(warning);
            }

            return {
              function: {
                name: tool.name,
                parameters,
                ...(tool.description ? { description: tool.description } : {}),
              },
              type: "function",
            };
          } else {
            warnings.push({
              details: "Only 'function' tool type is supported.",
              feature: `tool type for ${tool.name}`,
              type: "unsupported",
            });
            return null;
          }
        })
        .filter((t): t is AzureOpenAiChatCompletionTool => t !== null);
    }

    const modelParams: Record<string, unknown> = deepMerge(
      fmSettings.modelParams ?? {},
      sapOptions?.modelParams ?? {},
    );

    applyParameterOverrides(
      modelParams,
      options as Record<string, unknown>,
      sapOptions?.modelParams as Record<string, unknown> | undefined,
      fmSettings.modelParams as Record<string, unknown> | undefined,
      PARAM_MAPPINGS,
    );

    if (options.stopSequences && options.stopSequences.length > 0) {
      modelParams.stop = options.stopSequences;
    }

    validateModelParamsWithWarnings(
      {
        frequencyPenalty: options.frequencyPenalty,
        maxTokens: options.maxOutputTokens,
        presencePenalty: options.presencePenalty,
        temperature: options.temperature,
        topP: options.topP,
      },
      warnings,
    );

    // Map Vercel AI SDK toolChoice to SAP Foundation Models tool_choice
    const toolChoice = mapToolChoice(options.toolChoice);

    let responseFormat: AzureOpenAiChatCompletionParameters["response_format"];
    if (options.responseFormat?.type === "json") {
      responseFormat = options.responseFormat.schema
        ? {
            json_schema: {
              description: options.responseFormat.description,
              name: options.responseFormat.name ?? "response",
              schema: options.responseFormat.schema as Record<string, unknown>,
              strict: null,
            },
            type: "json_schema" as const,
          }
        : { type: "json_object" as const };
    } else if (fmSettings.responseFormat) {
      responseFormat =
        fmSettings.responseFormat as AzureOpenAiChatCompletionParameters["response_format"];
    }

    if (responseFormat && responseFormat.type !== "text") {
      warnings.push({
        message:
          "responseFormat JSON mode is forwarded to the underlying model; support and schema adherence depend on the model/deployment.",
        type: "other",
      });
    }

    // Pass through all model params (known and unknown) to the API
    // This allows users to send vendor-specific parameters
    const request: AzureOpenAiChatCompletionParameters = {
      messages: messages as AzureOpenAiChatCompletionParameters["messages"],
      ...modelParams,
      ...(tools && tools.length > 0 ? { tools } : {}),
      ...(toolChoice ? { tool_choice: toolChoice } : {}),
      ...(responseFormat ? { response_format: responseFormat } : {}),
      ...(fmSettings.dataSources &&
      Array.isArray(fmSettings.dataSources) &&
      fmSettings.dataSources.length > 0
        ? {
            data_sources:
              fmSettings.dataSources as AzureOpenAiChatCompletionParameters["data_sources"],
          }
        : {}),
    };

    return { request, warnings };
  }

  /**
   * Creates an SAP AI SDK AzureOpenAiChatClient with the given configuration.
   * @param config - The strategy configuration containing deployment info.
   * @param modelVersion - Optional model version for deployment resolution.
   * @returns A new AzureOpenAiChatClient instance.
   * @internal
   */
  private createClient(
    config: LanguageModelStrategyConfig,
    modelVersion?: string,
  ): InstanceType<AzureOpenAiChatClientClass> {
    const modelDeployment = buildModelDeployment(config, modelVersion);
    return new this.ClientClass(modelDeployment, config.destination);
  }
}
