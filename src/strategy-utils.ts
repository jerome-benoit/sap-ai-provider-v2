/**
 * Shared utilities for SAP AI Core strategy implementations.
 *
 * Contains common functions used by both Orchestration and Foundation Models strategies
 * to avoid code duplication and ensure consistency.
 */
import type {
  EmbeddingModelV3Embedding,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3FunctionTool,
  LanguageModelV3GenerateResult,
  LanguageModelV3StreamPart,
  SharedV3Warning,
} from "@ai-sdk/provider";
import type { DeploymentIdConfig, ResourceGroupConfig } from "@sap-ai-sdk/ai-api/internal.js";
import type { ZodType } from "zod";

import { z } from "zod";

import { deepMerge } from "./deep-merge.js";
import { validateModelParamsWithWarnings } from "./sap-ai-provider-options.js";

/**
 * Vercel AI SDK tool choice type.
 * @internal
 */
export type AISDKToolChoice =
  | { toolName: string; type: "tool" }
  | { type: "auto" }
  | { type: "none" }
  | { type: "required" };

/**
 * Base configuration for model deployment resolution.
 * Shared fields used by buildModelDeployment helper.
 * @internal
 */
export interface BaseModelDeploymentConfig {
  /** Deployment configuration (ID-based or resource group-based). */
  readonly deploymentConfig: DeploymentIdConfig | ResourceGroupConfig;
  /** The model identifier (e.g., 'gpt-4o', 'text-embedding-ada-002'). */
  readonly modelId: string;
}

/**
 * Configuration for building model parameters.
 * @internal
 */
export interface BuildModelParamsConfig {
  /** AI SDK call options. */
  readonly options: LanguageModelV3CallOptions;
  /** Parameter mappings for the strategy. */
  readonly paramMappings: readonly ParamMapping[];
  /** Provider options model params. */
  readonly providerModelParams?: Record<string, unknown>;
  /** Settings model params. */
  readonly settingsModelParams?: Record<string, unknown>;
}

/**
 * Result of building model parameters.
 * @internal
 */
export interface BuildModelParamsResult {
  /** The merged and validated model parameters. */
  readonly modelParams: Record<string, unknown>;
  /** Validation warnings. */
  readonly warnings: SharedV3Warning[];
}

/**
 * Result of converting AI SDK response format to SAP format.
 * @internal
 */
export interface ConvertedResponseFormatResult {
  /** The converted response format, or undefined if not applicable. */
  readonly responseFormat: SAPResponseFormat | undefined;
  /** Warning about JSON mode support, if applicable. */
  readonly warning: SharedV3Warning | undefined;
}

/**
 * Result of converting AI SDK tools to SAP format.
 * @internal
 */
export interface ConvertedToolsResult<T> {
  /** The converted tools array, or undefined if no tools. */
  readonly tools: T[] | undefined;
  /** Warnings generated during conversion. */
  readonly warnings: SharedV3Warning[];
}

/**
 * Result of extracting tool parameters from an AI SDK tool.
 * @internal
 */
export interface ExtractedToolParameters {
  /** The extracted SAP-compatible parameters. */
  readonly parameters: SAPToolParameters;
  /** Optional warning if schema conversion failed. */
  readonly warning?: SharedV3Warning;
}

/**
 * Extended function tool interface with optional parameters property.
 * @internal
 */
export interface FunctionToolWithParameters extends LanguageModelV3FunctionTool {
  readonly parameters?: unknown;
}

/**
 * Configuration for building a LanguageModelV3GenerateResult.
 * @internal
 */
export interface GenerateResultConfig {
  /** Model identifier (e.g., 'gpt-4o'). */
  readonly modelId: string;
  /** Provider name for providerMetadata key. */
  readonly providerName: string;
  /** Original request body for response.request.body. */
  readonly requestBody: unknown;
  /** SDK response object with accessor methods. */
  readonly response: SDKResponse;
  /** Normalized response headers. */
  readonly responseHeaders: Record<string, string> | undefined;
  /** Provider version string for metadata. */
  readonly version: string;
  /** Warnings from request building. */
  readonly warnings: SharedV3Warning[];
}

/**
 * Parameter mapping for AI SDK options → SAP model params.
 *
 * Used to map between different parameter naming conventions:
 * - AI SDK uses camelCase (e.g., `maxOutputTokens`)
 * - SAP APIs use snake_case (e.g., `max_tokens`)
 * @internal
 */
export interface ParamMapping {
  /** camelCase key in modelParams to read from and remove (e.g., 'maxTokens', 'topP'). */
  readonly camelCaseKey?: string;
  /** AI SDK option key (e.g., 'maxOutputTokens', 'topP'). */
  readonly optionKey?: string;
  /** Output key for SAP API (e.g., 'max_tokens', 'top_p'). */
  readonly outputKey: string;
}

/**
 * SAP-compatible response format type.
 * Used by both Orchestration and Foundation Models APIs.
 * @internal
 */
export type SAPResponseFormat =
  | {
      json_schema: {
        description?: string;
        name: string;
        schema: Record<string, unknown>;
        strict: boolean | null;
      };
      type: "json_schema";
    }
  | { type: "json_object" }
  | { type: "text" };

/**
 * SAP-compatible tool definition.
 * Common structure used by both APIs.
 * Uses generic parameter type to accommodate different SDK types.
 * @internal
 */
export interface SAPTool<P = SAPToolParameters> {
  function: {
    description?: string;
    name: string;
    parameters?: P;
  };
  type: "function";
}

/**
 * SAP Foundation Models SDK tool_choice type.
 * Matches AzureOpenAiChatCompletionToolChoiceOption from `@sap-ai-sdk/foundation-models`.
 * @internal
 */
export type SAPToolChoice =
  | "auto"
  | "none"
  | "required"
  | { function: { name: string }; type: "function" };

/**
 * SAP-compatible tool parameters structure.
 * Must have type "object" as required by the SAP AI APIs.
 * @internal
 */
export type SAPToolParameters = Record<string, unknown> & {
  type: "object";
};

/**
 * Common interface for SDK response objects (duck-typed).
 * Implemented by OrchestrationResponse and AzureOpenAiChatCompletionResponse.
 * @internal
 */
export interface SDKResponse {
  /** Returns text content. */
  getContent(): null | string | undefined;
  /** Returns finish reason string. */
  getFinishReason(): null | string | undefined;
  /** Returns token usage statistics. */
  getTokenUsage(): undefined | { completion_tokens?: number; prompt_tokens?: number };
  /** Returns tool calls array if present. */
  getToolCalls():
    | null
    | undefined
    | {
        function: { arguments: string; name: string };
        id: string;
      }[];
  /** Raw HTTP response with headers. */
  rawResponse: { headers: Headers | Record<string, string> };
}

/**
 * Interface for SDK stream chunks.
 * Both Orchestration and Foundation Models SDK chunks implement these methods.
 * @internal
 */
export interface SDKStreamChunk {
  /** Internal data for raw chunk emission. */
  _data?: unknown;
  /** Get the text content delta. */
  getDeltaContent(): null | string | undefined;
  /** Get the tool call deltas. */
  getDeltaToolCalls():
    | null
    | undefined
    | {
        function?: { arguments?: string; name?: string };
        id?: string;
        index?: number;
      }[];
  /** Get the finish reason if present in this chunk. */
  getFinishReason(): null | string | undefined;
}

/**
 * State object for tracking streaming response processing.
 * @internal
 */
export interface StreamState {
  /** Whether a text block is currently active. */
  activeText: boolean;
  /** The finish reason for the response. */
  finishReason: LanguageModelV3FinishReason;
  /** Whether this is the first chunk in the stream. */
  isFirstChunk: boolean;
  /** Token usage tracking. */
  usage: {
    inputTokens: {
      cacheRead: number | undefined;
      cacheWrite: number | undefined;
      noCache: number | undefined;
      total: number | undefined;
    };
    outputTokens: {
      reasoning: number | undefined;
      text: number | undefined;
      total: number | undefined;
    };
  };
}

/**
 * Configuration for creating a stream transformer.
 * @internal
 */
export interface StreamTransformerConfig {
  /** Function to convert errors to AI SDK format. */
  readonly convertToAISDKError: (
    error: unknown,
    context: { operation: string; requestBody: unknown; url: string },
  ) => unknown;
  /** The ID generator for creating unique IDs. */
  readonly idGenerator: StreamIdGenerator;
  /** Whether to include raw chunks in the output. */
  readonly includeRawChunks: boolean;
  /** The model identifier. */
  readonly modelId: string;
  /** The AI SDK call options (for error context). */
  readonly options: LanguageModelV3CallOptions;
  /** The provider name for metadata. */
  readonly providerName: string;
  /** The pre-generated response ID. */
  readonly responseId: string;
  /** The SDK stream to transform. */
  readonly sdkStream: AsyncIterable<SDKStreamChunk>;
  /** Function to get the final finish reason from the stream response. */
  readonly streamResponseGetFinishReason: () => null | string | undefined;
  /** Function to get the final token usage from the stream response. */
  readonly streamResponseGetTokenUsage: () =>
    | null
    | undefined
    | { completion_tokens?: number; prompt_tokens?: number };
  /** The URL identifier for error context. */
  readonly url: string;
  /** The provider version string for metadata. */
  readonly version: string;
  /** Warnings to include in the stream-start event. */
  readonly warnings: readonly SharedV3Warning[];
}

/**
 * Represents a tool call being accumulated during streaming.
 * @internal
 */
export interface ToolCallInProgress {
  /** Accumulated JSON arguments string. */
  arguments: string;
  /** Whether the tool-call event has been emitted. */
  didEmitCall: boolean;
  /** Whether the tool-input-start event has been emitted. */
  didEmitInputStart: boolean;
  /** The tool call identifier. */
  id: string;
  /** The name of the tool being called. */
  toolName?: string;
}

/**
 * AI SDK tool interface for conversion to SAP format.
 *
 * Represents the common structure of tools from Vercel AI SDK that can be
 * converted to SAP-compatible format. Only 'function' type tools are supported.
 * @internal
 */
interface AISDKTool {
  /** Optional tool description. */
  description?: string;
  /** JSON Schema for tool input parameters. */
  inputSchema?: unknown;
  /** Tool name identifier. */
  name: string;
  /** Tool type (only 'function' is supported). */
  type: string;
}

/**
 * Generates unique IDs for streaming response parts.
 *
 * Uses crypto.randomUUID() for cryptographically secure unique identifiers.
 * @internal
 */
export class StreamIdGenerator {
  /**
   * Generates a unique response ID.
   * @returns A UUID string for identifying the response.
   */
  generateResponseId(): string {
    return crypto.randomUUID();
  }

  /**
   * Generates a unique text block ID.
   * @returns A UUID string for identifying a text block.
   */
  generateTextBlockId(): string {
    return crypto.randomUUID();
  }
}

/**
 * Applies parameter overrides from AI SDK options and modelParams.
 * @param modelParams - The model parameters object to modify.
 * @param options - AI SDK call options.
 * @param sapModelParams - Provider options model params.
 * @param settingsModelParams - Settings model params.
 * @param mappings - Parameter mappings for this strategy.
 * @internal
 */
export function applyParameterOverrides(
  modelParams: Record<string, unknown>,
  options: Record<string, unknown>,
  sapModelParams: Record<string, unknown> | undefined,
  settingsModelParams: Record<string, unknown> | undefined,
  mappings: readonly ParamMapping[],
): void {
  for (const mapping of mappings) {
    const value =
      (mapping.optionKey ? options[mapping.optionKey] : undefined) ??
      (mapping.camelCaseKey ? sapModelParams?.[mapping.camelCaseKey] : undefined) ??
      (mapping.camelCaseKey ? settingsModelParams?.[mapping.camelCaseKey] : undefined);

    if (value !== undefined) {
      modelParams[mapping.outputKey] = value;
    }

    if (mapping.camelCaseKey && mapping.camelCaseKey !== mapping.outputKey) {
      // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
      delete modelParams[mapping.camelCaseKey];
    }
  }
}

/**
 * Builds a LanguageModelV3GenerateResult from SDK response.
 * @param config - Configuration with response and metadata.
 * @returns Complete generate result for AI SDK.
 * @internal
 */
export function buildGenerateResult(config: GenerateResultConfig): LanguageModelV3GenerateResult {
  const { modelId, providerName, requestBody, response, responseHeaders, version, warnings } =
    config;

  const content = extractResponseContent(response);

  const tokenUsage = response.getTokenUsage();
  const finishReasonRaw = response.getFinishReason();
  const finishReason = mapFinishReason(finishReasonRaw);

  const textContent = response.getContent();
  const toolCalls = response.getToolCalls();

  const rawResponseBody = {
    content: textContent,
    finishReason: finishReasonRaw,
    tokenUsage,
    toolCalls,
  };

  return {
    content,
    finishReason,
    providerMetadata: {
      [providerName]: {
        finishReason: finishReasonRaw ?? "unknown",
        finishReasonMapped: finishReason,
        ...(typeof responseHeaders?.["x-request-id"] === "string"
          ? { requestId: responseHeaders["x-request-id"] }
          : {}),
        version,
      },
    },
    request: {
      body: requestBody,
    },
    response: {
      body: rawResponseBody,
      headers: responseHeaders,
      modelId,
      timestamp: new Date(),
    },
    usage: {
      inputTokens: {
        cacheRead: undefined,
        cacheWrite: undefined,
        noCache: tokenUsage?.prompt_tokens,
        total: tokenUsage?.prompt_tokens,
      },
      outputTokens: {
        reasoning: undefined,
        text: tokenUsage?.completion_tokens,
        total: tokenUsage?.completion_tokens,
      },
    },
    warnings,
  };
}

/**
 * Builds a ModelDeployment object for the Foundation Models API SDK.
 *
 * Supports both deployment resolution strategies:
 * - Direct deploymentId: Uses the specific deployment directly
 * - Model-based: Uses modelName with optional modelVersion and resourceGroup
 * @param config - The strategy configuration containing deployment info and model ID.
 * @param modelVersion - Optional model version for model-based resolution.
 * @returns A ModelDeployment object for the Foundation Models API SDK.
 * @internal
 */
export function buildModelDeployment(
  config: BaseModelDeploymentConfig,
  modelVersion?: string,
): { deploymentId: string } | { modelName: string; modelVersion?: string; resourceGroup?: string } {
  const deploymentConfig = config.deploymentConfig;

  // Use deploymentId directly if provided
  if ("deploymentId" in deploymentConfig) {
    return { deploymentId: deploymentConfig.deploymentId };
  }

  // Build model-based deployment with optional version and resourceGroup
  const resourceGroup =
    "resourceGroup" in deploymentConfig ? deploymentConfig.resourceGroup : undefined;

  return {
    modelName: config.modelId,
    ...(modelVersion && { modelVersion }),
    ...(resourceGroup && { resourceGroup }),
  };
}

/**
 * Builds and validates model parameters from multiple sources.
 *
 * Merges parameters from settings and provider options, applies AI SDK option overrides,
 * handles stop sequences, and validates parameter ranges.
 * @param config - Configuration with options, mappings, and source parameters.
 * @returns The merged model parameters and validation warnings.
 * @internal
 */
export function buildModelParams(config: BuildModelParamsConfig): BuildModelParamsResult {
  const { options, paramMappings, providerModelParams, settingsModelParams } = config;
  const warnings: SharedV3Warning[] = [];

  // Deep merge settings and provider options
  const modelParams: Record<string, unknown> = deepMerge(
    settingsModelParams ?? {},
    providerModelParams ?? {},
  );

  // Apply parameter overrides from AI SDK options
  applyParameterOverrides(
    modelParams,
    options as Record<string, unknown>,
    providerModelParams,
    settingsModelParams,
    paramMappings,
  );

  // Handle stop sequences
  if (options.stopSequences && options.stopSequences.length > 0) {
    modelParams.stop = options.stopSequences;
  }

  // Validate parameter ranges
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

  return { modelParams, warnings };
}

/**
 * Builds SAP AI SDK-compatible tool parameters from a JSON schema.
 *
 * Handles edge cases:
 * - Non-object schemas are converted to empty object schemas
 * - Preserves additional schema fields (description, etc.)
 * - Validates properties and required arrays
 * @param schema - The JSON schema to convert.
 * @returns The SAP-compatible tool parameters object.
 * @internal
 */
export function buildSAPToolParameters(schema: Record<string, unknown>): SAPToolParameters {
  const schemaType = schema.type;

  // Non-object schemas are not supported - return empty object schema
  if (schemaType !== undefined && schemaType !== "object") {
    return {
      properties: {},
      required: [],
      type: "object",
    };
  }

  const properties =
    schema.properties && typeof schema.properties === "object"
      ? (schema.properties as Record<string, unknown>)
      : {};

  const required =
    Array.isArray(schema.required) && schema.required.every((item) => typeof item === "string")
      ? schema.required
      : [];

  // Preserve additional fields like description, additionalProperties, etc.
  const additionalFields = Object.fromEntries(
    Object.entries(schema).filter(
      ([key]) => key !== "type" && key !== "properties" && key !== "required",
    ),
  );

  return {
    properties,
    required,
    type: "object",
    ...additionalFields,
  };
}

/**
 * Converts AI SDK response format to SAP-compatible format.
 *
 * Handles conversion of structured output schemas:
 * - `{ type: 'json', schema: ... }` → `{ type: 'json_schema', json_schema: ... }`
 * - `{ type: 'json' }` → `{ type: 'json_object' }`
 * @param optionsResponseFormat - The AI SDK response format from call options.
 * @param settingsResponseFormat - The fallback response format from settings.
 * @returns The converted response format and any warning.
 * @internal
 */
export function convertResponseFormat(
  optionsResponseFormat: LanguageModelV3CallOptions["responseFormat"],
  settingsResponseFormat?: unknown,
): ConvertedResponseFormatResult {
  let responseFormat: SAPResponseFormat | undefined;
  let warning: SharedV3Warning | undefined;

  if (optionsResponseFormat?.type === "json") {
    responseFormat = optionsResponseFormat.schema
      ? {
          json_schema: {
            description: optionsResponseFormat.description,
            name: optionsResponseFormat.name ?? "response",
            schema: optionsResponseFormat.schema as Record<string, unknown>,
            strict: null,
          },
          type: "json_schema" as const,
        }
      : { type: "json_object" as const };
  } else if (settingsResponseFormat) {
    responseFormat = settingsResponseFormat as SAPResponseFormat;
  }

  if (responseFormat && responseFormat.type !== "text") {
    warning = {
      message:
        "responseFormat JSON mode is forwarded to the underlying model; support and schema adherence depend on the model/deployment.",
      type: "other",
    };
  }

  return { responseFormat, warning };
}

/**
 * Converts AI SDK tools to SAP-compatible tool format.
 *
 * This helper extracts the common tool conversion logic used by both
 * Orchestration and Foundation Models strategies.
 * @param tools - The AI SDK tools to convert.
 * @returns The converted tools and any warnings.
 * @template T - The specific SAP tool type (allows API-specific extensions).
 * @internal
 */
export function convertToolsToSAPFormat<T extends SAPTool<unknown>>(
  tools: AISDKTool[] | undefined,
): ConvertedToolsResult<T> {
  const warnings: SharedV3Warning[] = [];

  if (!tools || tools.length === 0) {
    return { tools: undefined, warnings };
  }

  const convertedTools = tools
    .map((tool): null | T => {
      if (tool.type === "function") {
        const { parameters, warning } = extractToolParameters(tool as LanguageModelV3FunctionTool);
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
        } as T;
      } else {
        warnings.push({
          details: "Only 'function' tool type is supported.",
          feature: `tool type for ${tool.name}`,
          type: "unsupported",
        });
        return null;
      }
    })
    .filter((t): t is T => t !== null);

  return {
    tools: convertedTools.length > 0 ? convertedTools : undefined,
    warnings,
  };
}

/**
 * Creates a summary of Vercel AI SDK request options for error context.
 *
 * Extracts key information without including sensitive prompt data.
 * Used for debugging and error reporting.
 * @param options - The language model call options to summarize.
 * @returns An object summarizing the request for debugging.
 * @internal
 */
export function createAISDKRequestBodySummary(options: LanguageModelV3CallOptions): {
  hasImageParts: boolean;
  maxOutputTokens?: number;
  promptMessages: number;
  responseFormatType?: string;
  seed?: number;
  stopSequences?: number;
  temperature?: number;
  toolChoiceType?: string;
  tools: number;
  topK?: number;
  topP?: number;
} {
  return {
    hasImageParts: options.prompt.some(
      (message) =>
        message.role === "user" &&
        message.content.some((part) => part.type === "file" && part.mediaType.startsWith("image/")),
    ),
    maxOutputTokens: options.maxOutputTokens,
    promptMessages: options.prompt.length,
    responseFormatType: options.responseFormat?.type,
    seed: options.seed,
    stopSequences: options.stopSequences?.length,
    temperature: options.temperature,
    toolChoiceType: options.toolChoice?.type,
    tools: options.tools?.length ?? 0,
    topK: options.topK,
    topP: options.topP,
  };
}

/**
 * Creates the initial stream state for processing streaming responses.
 *
 * Provides consistent initial state across both Orchestration and Foundation Models strategies.
 * @returns The initial stream state object.
 * @internal
 */
export function createInitialStreamState(): StreamState {
  return {
    activeText: false,
    finishReason: {
      raw: undefined,
      unified: "other" as const,
    },
    isFirstChunk: true,
    usage: {
      inputTokens: {
        cacheRead: undefined,
        cacheWrite: undefined,
        noCache: undefined,
        total: undefined,
      },
      outputTokens: {
        reasoning: undefined,
        text: undefined,
        total: undefined,
      },
    },
  };
}

/**
 * Creates a ReadableStream that transforms SAP AI SDK streaming responses
 * into Vercel AI SDK LanguageModelV3StreamPart events.
 *
 * This function encapsulates the common streaming logic used by both
 * Orchestration and Foundation Models strategies, handling:
 * - Stream lifecycle events (stream-start, response-metadata, finish)
 * - Text content streaming (text-start, text-delta, text-end)
 * - Tool call streaming (tool-input-start, tool-input-delta, tool-input-end, tool-call)
 * - Error handling and conversion to AI SDK errors
 * - Token usage extraction
 * @param config - The stream transformer configuration containing all dependencies.
 * @returns A ReadableStream of LanguageModelV3StreamPart events.
 * @internal
 */
export function createStreamTransformer(
  config: StreamTransformerConfig,
): ReadableStream<LanguageModelV3StreamPart> {
  const {
    convertToAISDKError,
    idGenerator,
    includeRawChunks,
    modelId,
    options,
    providerName,
    responseId,
    sdkStream,
    streamResponseGetFinishReason,
    streamResponseGetTokenUsage,
    url,
    version,
    warnings,
  } = config;

  let textBlockId: null | string = null;
  const streamState = createInitialStreamState();
  const toolCallsInProgress = new Map<number, ToolCallInProgress>();

  return new ReadableStream<LanguageModelV3StreamPart>({
    cancel() {
      // No cleanup needed - SDK handles stream cancellation internally
    },
    async start(controller) {
      controller.enqueue({
        type: "stream-start",
        warnings: [...warnings],
      });

      try {
        for await (const chunk of sdkStream) {
          if (includeRawChunks) {
            controller.enqueue({
              rawValue: (chunk as { _data?: unknown })._data ?? chunk,
              type: "raw",
            });
          }

          if (streamState.isFirstChunk) {
            streamState.isFirstChunk = false;
            controller.enqueue({
              id: responseId,
              modelId,
              timestamp: new Date(),
              type: "response-metadata",
            });
          }

          const deltaToolCalls = chunk.getDeltaToolCalls();
          if (Array.isArray(deltaToolCalls) && deltaToolCalls.length > 0) {
            streamState.finishReason = {
              raw: undefined,
              unified: "tool-calls",
            };
          }

          const deltaContent = chunk.getDeltaContent();
          if (
            typeof deltaContent === "string" &&
            deltaContent.length > 0 &&
            streamState.finishReason.unified !== "tool-calls"
          ) {
            if (!streamState.activeText) {
              textBlockId = idGenerator.generateTextBlockId();
              controller.enqueue({ id: textBlockId, type: "text-start" });
              streamState.activeText = true;
            }
            if (textBlockId) {
              controller.enqueue({
                delta: deltaContent,
                id: textBlockId,
                type: "text-delta",
              });
            }
          }

          if (Array.isArray(deltaToolCalls) && deltaToolCalls.length > 0) {
            for (const toolCallChunk of deltaToolCalls) {
              const index = toolCallChunk.index;
              if (typeof index !== "number" || !Number.isFinite(index)) {
                continue;
              }

              if (!toolCallsInProgress.has(index)) {
                toolCallsInProgress.set(index, {
                  arguments: "",
                  didEmitCall: false,
                  didEmitInputStart: false,
                  id: toolCallChunk.id ?? `tool_${String(index)}`,
                  toolName: toolCallChunk.function?.name,
                });
              }

              const tc = toolCallsInProgress.get(index);
              if (!tc) continue;

              if (toolCallChunk.id) {
                tc.id = toolCallChunk.id;
              }

              const nextToolName = toolCallChunk.function?.name;
              if (typeof nextToolName === "string" && nextToolName.length > 0) {
                tc.toolName = nextToolName;
              }

              if (!tc.didEmitInputStart && tc.toolName) {
                tc.didEmitInputStart = true;
                controller.enqueue({
                  id: tc.id,
                  toolName: tc.toolName,
                  type: "tool-input-start",
                });
              }

              const argumentsDelta = toolCallChunk.function?.arguments;
              if (typeof argumentsDelta === "string" && argumentsDelta.length > 0) {
                tc.arguments += argumentsDelta;

                if (tc.didEmitInputStart) {
                  controller.enqueue({
                    delta: argumentsDelta,
                    id: tc.id,
                    type: "tool-input-delta",
                  });
                }
              }
            }
          }

          const chunkFinishReason = chunk.getFinishReason();
          if (chunkFinishReason) {
            streamState.finishReason = mapFinishReason(chunkFinishReason);

            if (streamState.finishReason.unified === "tool-calls") {
              const toolCalls = Array.from(toolCallsInProgress.values());
              for (const tc of toolCalls) {
                if (tc.didEmitCall) {
                  continue;
                }
                if (!tc.didEmitInputStart) {
                  tc.didEmitInputStart = true;
                  controller.enqueue({
                    id: tc.id,
                    toolName: tc.toolName ?? "",
                    type: "tool-input-start",
                  });
                }

                tc.didEmitCall = true;
                controller.enqueue({ id: tc.id, type: "tool-input-end" });
                controller.enqueue({
                  input: tc.arguments,
                  toolCallId: tc.id,
                  toolName: tc.toolName ?? "",
                  type: "tool-call",
                });
              }

              if (streamState.activeText && textBlockId) {
                controller.enqueue({ id: textBlockId, type: "text-end" });
                streamState.activeText = false;
              }
            }
          }
        }

        const toolCalls = Array.from(toolCallsInProgress.values());
        let didEmitAnyToolCalls = false;

        for (const tc of toolCalls) {
          if (tc.didEmitCall) {
            continue;
          }

          if (!tc.didEmitInputStart) {
            tc.didEmitInputStart = true;
            controller.enqueue({
              id: tc.id,
              toolName: tc.toolName ?? "",
              type: "tool-input-start",
            });
          }

          didEmitAnyToolCalls = true;
          tc.didEmitCall = true;
          controller.enqueue({ id: tc.id, type: "tool-input-end" });
          controller.enqueue({
            input: tc.arguments,
            toolCallId: tc.id,
            toolName: tc.toolName ?? "",
            type: "tool-call",
          });
        }

        if (streamState.activeText && textBlockId) {
          controller.enqueue({ id: textBlockId, type: "text-end" });
        }

        const finalFinishReason = streamResponseGetFinishReason();
        if (finalFinishReason) {
          streamState.finishReason = mapFinishReason(finalFinishReason);
        } else if (didEmitAnyToolCalls) {
          streamState.finishReason = {
            raw: undefined,
            unified: "tool-calls",
          };
        }

        const finalUsage = streamResponseGetTokenUsage();
        if (finalUsage) {
          streamState.usage.inputTokens.total = finalUsage.prompt_tokens;
          streamState.usage.inputTokens.noCache = finalUsage.prompt_tokens;
          streamState.usage.outputTokens.total = finalUsage.completion_tokens;
          streamState.usage.outputTokens.text = finalUsage.completion_tokens;
        }

        controller.enqueue({
          finishReason: streamState.finishReason,
          providerMetadata: {
            [providerName]: {
              finishReason: streamState.finishReason.raw,
              responseId,
              version,
            },
          },
          type: "finish",
          usage: streamState.usage,
        });

        controller.close();
      } catch (error) {
        const aiError = convertToAISDKError(error, {
          operation: "doStream",
          requestBody: createAISDKRequestBodySummary(options),
          url,
        });
        controller.enqueue({
          error: aiError instanceof Error ? aiError : new Error(String(aiError)),
          type: "error",
        });
        controller.close();
      }
    },
  });
}

/**
 * Extracts content (text and tool calls) from SDK response.
 * @param response - SDK response object.
 * @returns Content array for LanguageModelV3GenerateResult.
 * @internal
 */
export function extractResponseContent(response: SDKResponse): LanguageModelV3Content[] {
  const content: LanguageModelV3Content[] = [];

  const textContent = response.getContent();
  if (textContent) {
    content.push({
      text: textContent,
      type: "text",
    });
  }

  const toolCalls = response.getToolCalls();
  if (toolCalls) {
    for (const toolCall of toolCalls) {
      content.push({
        input: toolCall.function.arguments,
        toolCallId: toolCall.id,
        toolName: toolCall.function.name,
        type: "tool-call",
      });
    }
  }

  return content;
}

/**
 * Extracts SAP-compatible tool parameters from an AI SDK function tool.
 *
 * Handles multiple schema formats:
 * - Zod schemas (converted via z.toJSONSchema)
 * - JSON Schema objects with properties
 * - Empty/missing schemas (returns empty object schema)
 * @param tool - The AI SDK function tool to extract parameters from.
 * @returns The extracted parameters and optional warning.
 * @internal
 */
export function extractToolParameters(tool: LanguageModelV3FunctionTool): ExtractedToolParameters {
  const inputSchema = tool.inputSchema as Record<string, unknown> | undefined;
  const toolWithParams = tool as FunctionToolWithParameters;

  if (toolWithParams.parameters && isZodSchema(toolWithParams.parameters)) {
    try {
      const jsonSchema = z.toJSONSchema(toolWithParams.parameters);
      const schemaRecord = jsonSchema as Record<string, unknown>;
      delete schemaRecord.$schema;
      return { parameters: buildSAPToolParameters(schemaRecord) };
    } catch (error) {
      return {
        parameters: buildSAPToolParameters({}),
        warning: {
          details: `Failed to convert tool Zod schema: ${error instanceof Error ? error.message : String(error)}. Falling back to empty object schema.`,
          feature: `tool schema conversion for ${tool.name}`,
          type: "unsupported",
        },
      };
    }
  }

  if (inputSchema && Object.keys(inputSchema).length > 0) {
    const hasProperties =
      inputSchema.properties &&
      typeof inputSchema.properties === "object" &&
      Object.keys(inputSchema.properties).length > 0;

    if (hasProperties) {
      return { parameters: buildSAPToolParameters(inputSchema) };
    }
  }

  return { parameters: buildSAPToolParameters({}) };
}

/**
 * Checks if an object has a callable parse method.
 * @param obj - The object to check for a parse method.
 * @returns True if the object has a callable parse method.
 * @internal
 */
export function hasCallableParse(
  obj: Record<string, unknown>,
): obj is Record<string, unknown> & { parse: (...args: unknown[]) => unknown } {
  return typeof obj.parse === "function";
}

/**
 * Type guard for Zod schema objects.
 *
 * Detects Zod schemas by checking for the presence of `_def` and a callable `parse` method.
 * @param obj - The object to check.
 * @returns True if the object is a Zod schema.
 * @internal
 */
export function isZodSchema(obj: unknown): obj is ZodType {
  if (obj === null || typeof obj !== "object") {
    return false;
  }
  const record = obj as Record<string, unknown>;
  return "_def" in record && "parse" in record && hasCallableParse(record);
}

/**
 * Maps provider finish reasons to Vercel AI SDK LanguageModelV3FinishReason.
 *
 * Handles various finish reason formats from different model providers:
 * - OpenAI: "stop", "length", "tool_calls", "content_filter"
 * - Anthropic: "end_turn", "stop_sequence", "max_tokens"
 * - Amazon: "eos", "max_tokens_reached"
 * - Others: "error", "function_call", "tool_call"
 * @param reason - The raw finish reason string from the provider.
 * @returns The unified finish reason with both raw and unified representations.
 * @internal
 */
export function mapFinishReason(reason: null | string | undefined): LanguageModelV3FinishReason {
  const raw = reason ?? undefined;

  if (!reason) {
    return { raw, unified: "other" };
  }

  switch (reason.toLowerCase()) {
    case "content_filter":
      return { raw, unified: "content-filter" };
    case "end_turn":
    case "eos":
    case "stop":
    case "stop_sequence":
      return { raw, unified: "stop" };
    case "error":
      return { raw, unified: "error" };
    case "function_call":
    case "tool_call":
    case "tool_calls":
      return { raw, unified: "tool-calls" };
    case "length":
    case "max_tokens":
    case "max_tokens_reached":
      return { raw, unified: "length" };
    default:
      return { raw, unified: "other" };
  }
}

/**
 * Maps Vercel AI SDK toolChoice to SAP Foundation Models SDK tool_choice format.
 *
 * Mapping:
 * - `{ type: 'auto' }` → `'auto'`
 * - `{ type: 'none' }` → `'none'`
 * - `{ type: 'required' }` → `'required'`
 * - `{ type: 'tool', toolName: 'fn' }` → `{ type: 'function', function: { name: 'fn' } }`
 * @param toolChoice - The Vercel AI SDK tool choice.
 * @returns The SAP SDK tool_choice format, or undefined if no mapping needed.
 * @internal
 */
export function mapToolChoice(toolChoice: AISDKToolChoice | undefined): SAPToolChoice | undefined {
  if (!toolChoice) {
    return undefined;
  }

  switch (toolChoice.type) {
    case "auto":
      return "auto";
    case "none":
      return "none";
    case "required":
      return "required";
    case "tool":
      return {
        function: { name: toolChoice.toolName },
        type: "function",
      };
    default:
      return undefined;
  }
}

/**
 * Converts SAP AI SDK embedding (number[] or base64) to Vercel AI SDK format.
 *
 * Handles both formats that can be returned by embedding APIs:
 * - Direct number arrays (most common)
 * - Base64-encoded float32 arrays (for bandwidth efficiency)
 * @param embedding - The embedding as number array or base64 string.
 * @returns The normalized embedding as a number array.
 * @internal
 */
export function normalizeEmbedding(embedding: number[] | string): EmbeddingModelV3Embedding {
  if (Array.isArray(embedding)) {
    return embedding;
  }
  // Base64-encoded float32 values
  const buffer = Buffer.from(embedding, "base64");
  const float32Array = new Float32Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.length / Float32Array.BYTES_PER_ELEMENT,
  );
  return Array.from(float32Array);
}
