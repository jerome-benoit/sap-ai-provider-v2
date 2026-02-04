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
import { convertToAISDKError, normalizeHeaders } from "./sap-ai-error.js";
import { getProviderName, sapAILanguageModelProviderOptions } from "./sap-ai-provider-options.js";
import {
  buildGenerateResult,
  buildModelDeployment,
  buildModelParams,
  convertResponseFormat,
  convertToolsToSAPFormat,
  createAISDKRequestBodySummary,
  createStreamTransformer,
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

    // Convert AI SDK tools to SAP format using shared helper
    const toolsResult = convertToolsToSAPFormat<AzureOpenAiChatCompletionTool>(options.tools);
    const tools = toolsResult.tools;
    warnings.push(...toolsResult.warnings);

    // Build model parameters using shared helper
    const { modelParams, warnings: paramWarnings } = buildModelParams({
      options,
      paramMappings: PARAM_MAPPINGS,
      providerModelParams: sapOptions?.modelParams as Record<string, unknown> | undefined,
      settingsModelParams: fmSettings.modelParams as Record<string, unknown> | undefined,
    });
    warnings.push(...paramWarnings);

    // Map Vercel AI SDK toolChoice to SAP Foundation Models tool_choice
    const toolChoice = mapToolChoice(options.toolChoice);

    // Convert response format using shared helper
    const { responseFormat, warning: responseFormatWarning } = convertResponseFormat(
      options.responseFormat,
      fmSettings.responseFormat,
    );
    if (responseFormatWarning) {
      warnings.push(responseFormatWarning);
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
