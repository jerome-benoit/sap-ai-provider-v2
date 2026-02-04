import type { AzureOpenAiChatCompletionParameters } from "@sap-ai-sdk/foundation-models";
import type {
  ChatCompletionTool,
  ChatModel,
  FilteringModule,
  GroundingModule,
  MaskingModule,
  TranslationModule,
} from "@sap-ai-sdk/orchestration";

/** Azure OpenAI chat extension configuration for "On Your Data" RAG scenarios. */
export type AzureOpenAiChatExtensionConfiguration = NonNullable<
  AzureOpenAiChatCompletionParameters["data_sources"]
>[number];

/** Common model parameters shared between both APIs. */
export interface CommonModelParams {
  readonly frequencyPenalty?: number;
  readonly maxTokens?: number;
  /** Not supported by Amazon/Anthropic models. */
  readonly n?: number;
  readonly parallel_tool_calls?: boolean;
  readonly presencePenalty?: number;
  readonly [key: string]: unknown;
  readonly temperature?: number;
  readonly topP?: number;
}

/** Default settings configuration when using Foundation Models API. */
export interface FoundationModelsDefaultSettings {
  readonly api: "foundation-models";
  readonly settings?: FoundationModelsModelSettings;
}

/** Model parameters for Foundation Models Embedding API. */
export interface FoundationModelsEmbeddingParams {
  readonly dimensions?: number;
  readonly encoding_format?: "base64" | "float";
  readonly user?: string;
}

/** Model parameters for Foundation Models API with Azure OpenAI-specific extensions. */
export interface FoundationModelsModelParams extends CommonModelParams {
  readonly logit_bias?: AzureOpenAiChatCompletionParameters["logit_bias"];
  readonly logprobs?: AzureOpenAiChatCompletionParameters["logprobs"];
  readonly seed?: AzureOpenAiChatCompletionParameters["seed"];
  readonly stop?: AzureOpenAiChatCompletionParameters["stop"];
  /** Requires logprobs=true. */
  readonly top_logprobs?: AzureOpenAiChatCompletionParameters["top_logprobs"];
  readonly user?: AzureOpenAiChatCompletionParameters["user"];
}

/** Model settings when using Foundation Models API. */
export interface FoundationModelsModelSettings {
  readonly api: "foundation-models";
  /** Azure OpenAI "On Your Data" configuration for RAG scenarios. */
  readonly dataSources?: AzureOpenAiChatExtensionConfiguration[];
  /** @default false */
  readonly includeReasoning?: boolean;
  readonly modelParams?: FoundationModelsModelParams;
  readonly modelVersion?: string;
  readonly responseFormat?: ResponseFormat;
}

/** Default settings configuration when using Orchestration API. */
export interface OrchestrationDefaultSettings {
  readonly api?: "orchestration";
  readonly settings?: OrchestrationModelSettings;
}

/** Model parameters for Orchestration API. */
export type OrchestrationModelParams = CommonModelParams;

/** Model settings when using Orchestration API with filtering, grounding, masking, and translation. */
export interface OrchestrationModelSettings {
  readonly api?: "orchestration";
  /** @default true */
  readonly escapeTemplatePlaceholders?: boolean;
  readonly filtering?: FilteringModule;
  readonly grounding?: GroundingModule;
  /** @default false */
  readonly includeReasoning?: boolean;
  readonly masking?: MaskingModule;
  readonly modelParams?: OrchestrationModelParams;
  readonly modelVersion?: string;
  readonly placeholderValues?: Record<string, string>;
  readonly promptTemplateRef?: PromptTemplateRef;
  readonly responseFormat?: ResponseFormat;
  readonly tools?: ChatCompletionTool[];
  readonly translation?: TranslationModule;
}

/** Reference to a template in SAP AI Core's Prompt Registry. */
export type PromptTemplateRef = PromptTemplateRefByID | PromptTemplateRefByScenarioNameVersion;

/** Reference to a Prompt Registry template by ID. */
export interface PromptTemplateRefByID {
  readonly id: string;
  readonly scope?: PromptTemplateScope;
}

/** Reference to a Prompt Registry template by scenario, name, and version. */
export interface PromptTemplateRefByScenarioNameVersion {
  readonly name: string;
  readonly scenario: string;
  readonly scope?: PromptTemplateScope;
  readonly version: string;
}

/** Scope for Prompt Registry templates: 'tenant' (default) or 'resource_group'. */
export type PromptTemplateScope = "resource_group" | "tenant";

/** Response format for structured output (OpenAI-compatible). */
export type ResponseFormat = AzureOpenAiChatCompletionParameters["response_format"];

/** Supported API types for SAP AI Core. */
export type SAPAIApiType = "foundation-models" | "orchestration";

/** Union type for API-specific default settings configuration. */
export type SAPAIDefaultSettingsConfig =
  | FoundationModelsDefaultSettings
  | OrchestrationDefaultSettings;

/** Settings for the SAP AI Embedding Model. */
export interface SAPAIEmbeddingSettings {
  readonly api?: SAPAIApiType;
  /** Orchestration API only. */
  readonly masking?: MaskingModule;
  /** @default 2048 */
  readonly maxEmbeddingsPerCall?: number;
  readonly modelParams?: FoundationModelsEmbeddingParams | Record<string, unknown>;
  readonly modelVersion?: string;
  readonly [key: string]: unknown;
  /** @default 'text' */
  readonly type?: "document" | "query" | "text";
}

/** Supported model IDs in SAP AI Core (availability depends on tenant configuration). */
export type SAPAIModelId = ChatModel;

/** Union type for model settings - supports both APIs. */
export type SAPAIModelSettings = FoundationModelsModelSettings | OrchestrationModelSettings;

/** Re-exported Azure OpenAI types from `@sap-ai-sdk/foundation-models`. */
export type {
  AzureOpenAiChatCompletionParameters,
  AzureOpenAiChatCompletionRequestAssistantMessage,
  AzureOpenAiChatCompletionRequestMessage,
  AzureOpenAiChatCompletionRequestSystemMessage,
  AzureOpenAiChatCompletionRequestToolMessage,
  AzureOpenAiChatCompletionRequestUserMessage,
  AzureOpenAiChatCompletionTool,
  AzureOpenAiEmbeddingParameters,
  AzureOpenAiFunctionObject,
} from "@sap-ai-sdk/foundation-models";

/** Re-exported SAP AI SDK orchestration types. */
export type {
  FilteringModule,
  GroundingModule,
  MaskingModule,
  TranslationModule,
} from "@sap-ai-sdk/orchestration";

export {
  buildAzureContentSafetyFilter,
  buildDocumentGroundingConfig,
  buildDpiMaskingProvider,
  buildLlamaGuard38BFilter,
  buildTranslationConfig,
} from "@sap-ai-sdk/orchestration";

/**
 * Settings for configuring SAP AI Core model behavior.
 * Legacy interface maintained for backward compatibility - prefer OrchestrationModelSettings
 * or FoundationModelsModelSettings for API-specific type safety.
 */
export interface SAPAISettings {
  readonly api?: SAPAIApiType;
  /** @default true */
  readonly escapeTemplatePlaceholders?: boolean;
  readonly filtering?: FilteringModule;
  readonly grounding?: GroundingModule;
  /** @default false */
  readonly includeReasoning?: boolean;
  readonly masking?: MaskingModule;
  readonly modelParams?: CommonModelParams;
  readonly modelVersion?: string;
  /** Orchestration API only. */
  readonly promptTemplateRef?: PromptTemplateRef;
  readonly responseFormat?: ResponseFormat;
  readonly tools?: ChatCompletionTool[];
  readonly translation?: TranslationModule;
}

export type {
  AssistantChatMessage,
  ChatCompletionRequest,
  ChatCompletionTool,
  ChatMessage,
  DeveloperChatMessage,
  DocumentTranslationApplyToSelector,
  FunctionObject,
  LlmModelDetails,
  LlmModelParams,
  OrchestrationConfigRef,
  OrchestrationModuleConfig,
  PromptTemplatingModule,
  SystemChatMessage,
  ToolChatMessage,
  TranslationApplyToCategory,
  TranslationInputParameters,
  TranslationOutputParameters,
  TranslationTargetLanguage,
  UserChatMessage,
} from "@sap-ai-sdk/orchestration";

export {
  OrchestrationEmbeddingResponse,
  OrchestrationResponse,
  OrchestrationStream,
  OrchestrationStreamChunkResponse,
  OrchestrationStreamResponse,
} from "@sap-ai-sdk/orchestration";
