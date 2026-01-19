/**
 * Adapters to convert Vercel AI SDK LanguageModelV3 formats to LanguageModelV2 formats.
 *
 * This module provides transformation functions that convert V3 response formats
 * (used internally by the SAP AI Core implementation) to V2 formats
 * (exposed as the public API).
 * @module sap-ai-adapters-v3-to-v2
 * @internal
 */

import type {
  LanguageModelV2CallWarning,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Warning,
} from "@ai-sdk/provider";

/**
 * Converts a V3 finish reason to V2 format.
 *
 * V3 format: `{ unified: string, raw?: string }`
 * V2 format: `string` (one of: 'stop', 'length', 'content-filter', 'tool-calls', 'error', 'other', 'unknown')
 * @param v3FinishReason - The V3 finish reason object
 * @returns The V2 finish reason string
 * @internal
 */
export function convertFinishReasonV3ToV2(
  v3FinishReason: LanguageModelV3FinishReason,
): LanguageModelV2FinishReason {
  return v3FinishReason.unified;
}

/**
 * Converts a V3 stream part to V2 format.
 *
 * V3 stream events have more granular structure:
 * - `text-start`, `text-delta`, `text-end` (with block IDs)
 * - `tool-input-start`, `tool-input-delta`, `tool-input-end`
 * - `finish` event with structured usage
 * - `stream-start` event with warnings
 *
 * V2 has the same event structure, so most events pass through unchanged.
 * The key transformations are:
 * - `finish` event: convert usage format
 * - `stream-start` event: convert warnings
 * @param v3Part - The V3 stream part
 * @returns The V2 stream part
 * @internal
 */
export function convertStreamPartV3ToV2(
  v3Part: LanguageModelV3StreamPart,
): LanguageModelV2StreamPart {
  switch (v3Part.type) {
    case "finish":
      return {
        finishReason: convertFinishReasonV3ToV2(v3Part.finishReason),
        // Cast providerMetadata as compatible between V2/V3
        // The type difference is only in index signature strictness
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
        providerMetadata: v3Part.providerMetadata as any,
        type: "finish",
        usage: convertUsageV3ToV2(v3Part.usage),
      };

    case "stream-start":
      return {
        type: "stream-start",
        warnings: convertWarningsV3ToV2(v3Part.warnings),
      };

    // All other event types are compatible between V2 and V3
    default:
      return v3Part as LanguageModelV2StreamPart;
  }
}

/**
 * Transforms a V3 stream to a V2 stream.
 *
 * This async generator function reads from a V3 ReadableStream and yields
 * V2-formatted stream parts, converting usage and finish reason formats on the fly.
 * @param v3Stream - The V3 ReadableStream
 * @yields {LanguageModelV2StreamPart} V2-formatted stream parts
 * @returns An async generator yielding V2 stream parts
 * @internal
 */
export async function* convertStreamV3ToV2(
  v3Stream: ReadableStream<LanguageModelV3StreamPart>,
): AsyncGenerator<LanguageModelV2StreamPart> {
  const reader = v3Stream.getReader();

  try {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        break;
      }

      yield convertStreamPartV3ToV2(value);
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Converts V3 usage information to V2 format.
 *
 * V3 format (nested):
 * ```
 * {
 *   inputTokens: { total, cacheRead, cacheWrite, noCache },
 *   outputTokens: { total, reasoning, text }
 * }
 * ```
 *
 * V2 format (flat):
 * ```
 * {
 *   inputTokens: number,
 *   outputTokens: number,
 *   totalTokens?: number,
 *   reasoningTokens?: number,
 *   cachedInputTokens?: number
 * }
 * ```
 * @param v3Usage - The V3 usage object
 * @returns The V2 usage object
 * @internal
 */
export function convertUsageV3ToV2(v3Usage: LanguageModelV3Usage): LanguageModelV2Usage {
  return {
    cachedInputTokens: v3Usage.inputTokens.cacheRead,
    inputTokens: v3Usage.inputTokens.total,
    outputTokens: v3Usage.outputTokens.total,
    reasoningTokens: v3Usage.outputTokens.reasoning,
    totalTokens:
      v3Usage.inputTokens.total !== undefined && v3Usage.outputTokens.total !== undefined
        ? v3Usage.inputTokens.total + v3Usage.outputTokens.total
        : undefined,
  };
}

/**
 * Converts an array of V3 warnings to V2 warnings.
 * @param v3Warnings - Array of V3 warning objects
 * @returns Array of V2 warning objects
 * @internal
 */
export function convertWarningsV3ToV2(v3Warnings: SharedV3Warning[]): LanguageModelV2CallWarning[] {
  return v3Warnings.map(convertWarningV3ToV2);
}

/**
 * Converts a V3 warning to V2 warning format.
 *
 * V3 warnings have a `feature` field for unsupported features.
 * V2 warnings use different types: 'unsupported-setting', 'unsupported-tool', or 'other'.
 *
 * Since we don't have detailed context about which setting/tool is unsupported,
 * we map V3 unsupported warnings to V2 'other' type with a descriptive message.
 * @param v3Warning - The V3 warning object
 * @returns The V2 warning object
 * @internal
 */
export function convertWarningV3ToV2(v3Warning: SharedV3Warning): LanguageModelV2CallWarning {
  if (v3Warning.type === "unsupported") {
    return {
      message: v3Warning.details
        ? `Unsupported feature: ${v3Warning.feature}. ${v3Warning.details}`
        : `Unsupported feature: ${v3Warning.feature}`,
      type: "other",
    };
  }

  if (v3Warning.type === "compatibility") {
    return {
      message: v3Warning.details
        ? `Compatibility mode: ${v3Warning.feature}. ${v3Warning.details}`
        : `Compatibility mode: ${v3Warning.feature}`,
      type: "other",
    };
  }

  // V3 'other' type maps directly to V2 'other' type
  return {
    message: v3Warning.message,
    type: "other",
  };
}
