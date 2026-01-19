/**
 * Unit tests for SAP AI Embedding Model V2
 *
 * Tests the V2 facade for embedding models, verifying that it correctly
 * delegates to the V3 implementation and transforms responses to V2 format.
 */

import { describe, expect, it, vi } from "vitest";

import { SAPAIEmbeddingModelV2 } from "./sap-ai-embedding-model-v2";

describe("SAPAIEmbeddingModelV2", () => {
  const defaultConfig = {
    deploymentConfig: { resourceGroup: "default" },
    provider: "sap-ai" as const,
  };

  describe("Constructor", () => {
    it("should create V2 embedding model with correct specification version", () => {
      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      expect(model.specificationVersion).toBe("v2");
      expect(model.provider).toBe("sap-ai");
      expect(model.modelId).toBe("text-embedding-ada-002");
    });

    it("should pass settings to V3 model", () => {
      const settings = {
        maxEmbeddingsPerCall: 1000,
        type: "query" as const,
      };

      const model = new SAPAIEmbeddingModelV2("text-embedding-3-small", settings, defaultConfig);

      expect(model.maxEmbeddingsPerCall).toBe(1000);
      expect(model.supportsParallelCalls).toBe(true);
    });
  });

  describe("doEmbed", () => {
    it("should delegate to V3 model and transform response", async () => {
      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      // Mock the V3 model's doEmbed method
      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [
          [0.1, 0.2, 0.3],
          [0.4, 0.5, 0.6],
        ],
        providerMetadata: {
          "sap-ai": { modelId: "text-embedding-ada-002" },
        },
        response: {
          body: { data: "test" },
          headers: { "x-request-id": "123" },
        },
        usage: { tokens: 10 },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      const result = await model.doEmbed({
        values: ["Hello", "World"],
      });

      expect(mockDoEmbed).toHaveBeenCalledWith({
        values: ["Hello", "World"],
      });

      expect(result.embeddings).toEqual([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
      ]);
      expect(result.usage).toEqual({ tokens: 10 });
      expect(result.providerMetadata).toBeDefined();
      expect(result.response?.headers).toBeDefined();
      expect(result.response?.body).toEqual({ data: "test" });
    });

    it("should handle warnings by logging them", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2, 0.3]],
        usage: { tokens: 5 },
        warnings: [
          {
            feature: "dimension",
            type: "unsupported" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      const result = await model.doEmbed({
        values: ["Test"],
      });

      expect(result.embeddings).toEqual([[0.1, 0.2, 0.3]]);
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        "[SAP AI Embedding] Unsupported feature: dimension",
      );

      consoleWarnSpy.mockRestore();
    });

    it("should handle warnings with details", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2]],
        usage: { tokens: 3 },
        warnings: [
          {
            details: "Custom dimensions not supported",
            feature: "dimensions parameter",
            type: "unsupported" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        values: ["Test"],
      });

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        "[SAP AI Embedding] Unsupported feature: dimensions parameter. Custom dimensions not supported",
      );

      consoleWarnSpy.mockRestore();
    });

    it("should handle empty warnings array", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2, 0.3]],
        usage: { tokens: 5 },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        values: ["Test"],
      });

      expect(consoleWarnSpy).not.toHaveBeenCalled();

      consoleWarnSpy.mockRestore();
    });

    it("should handle unsupported-setting warnings (type coverage)", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2]],
        usage: { tokens: 3 },
        warnings: [
          {
            details: "This setting is not supported",
            feature: "custom-setting",
            setting: "maxRetries",
            type: "unsupported" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        values: ["Test"],
      });

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        "[SAP AI Embedding] Unsupported feature: custom-setting. This setting is not supported",
      );

      consoleWarnSpy.mockRestore();
    });

    it("should handle other warning type (type coverage)", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2]],
        usage: { tokens: 3 },
        warnings: [
          {
            message: "General warning message",
            type: "other" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        values: ["Test"],
      });

      expect(consoleWarnSpy).toHaveBeenCalledWith("[SAP AI Embedding] General warning message");

      consoleWarnSpy.mockRestore();
    });

    it("should handle compatibility warnings (maps to other type)", async () => {
      const consoleWarnSpy = vi.spyOn(console, "warn").mockImplementation(vi.fn());

      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1, 0.2]],
        usage: { tokens: 3 },
        warnings: [
          {
            details: "Using compatibility mode",
            feature: "legacy-api",
            type: "compatibility" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        values: ["Test"],
      });

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        "[SAP AI Embedding] Compatibility mode: legacy-api. Using compatibility mode",
      );

      consoleWarnSpy.mockRestore();
    });

    it("should pass through abort signal", async () => {
      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const abortController = new AbortController();
      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1]],
        usage: { tokens: 1 },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        abortSignal: abortController.signal,
        values: ["Test"],
      });

      expect(mockDoEmbed).toHaveBeenCalledWith({
        abortSignal: abortController.signal,
        values: ["Test"],
      });
    });

    it("should pass through provider options", async () => {
      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const providerOptions = { "sap-ai": { type: "query" } };
      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1]],
        usage: { tokens: 1 },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        providerOptions,
        values: ["Test"],
      });

      expect(mockDoEmbed).toHaveBeenCalledWith({
        providerOptions,
        values: ["Test"],
      });
    });

    it("should pass through headers", async () => {
      const model = new SAPAIEmbeddingModelV2("text-embedding-ada-002", {}, defaultConfig);

      const headers = { "X-Custom-Header": "test-value" };
      const mockDoEmbed = vi.fn().mockResolvedValue({
        embeddings: [[0.1]],
        usage: { tokens: 1 },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doEmbed = mockDoEmbed;

      await model.doEmbed({
        headers,
        values: ["Test"],
      });

      expect(mockDoEmbed).toHaveBeenCalledWith({
        headers,
        values: ["Test"],
      });
    });
  });
});
