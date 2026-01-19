/**
 * Unit tests for SAP AI Language Model V2
 *
 * Tests the V2 facade for language models, verifying that it correctly
 * delegates to the V3 implementation and transforms responses to V2 format.
 */

import { describe, expect, it, vi } from "vitest";

import { SAPAILanguageModelV2 } from "./sap-ai-language-model-v2";

describe("SAPAILanguageModelV2", () => {
  const defaultConfig = {
    deploymentConfig: { resourceGroup: "default" },
    provider: "sap-ai" as const,
  };

  describe("Constructor and Properties", () => {
    it("should create V2 language model with correct specification version", () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      expect(model.specificationVersion).toBe("v2");
      expect(model.provider).toBe("sap-ai");
      expect(model.modelId).toBe("gpt-4o");
    });

    it("should delegate provider property to V3 model", () => {
      const model = new SAPAILanguageModelV2("gpt-4o-mini", {}, defaultConfig);

      expect(model.provider).toBe("sap-ai");
    });

    it("should delegate supportedUrls property to V3 model", () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      expect(model.supportedUrls).toBeDefined();
      expect(typeof model.supportedUrls).toBe("object");
    });

    it("should pass settings to V3 model", () => {
      const settings = {
        modelParams: {
          temperature: 0.7,
        },
      };

      const model = new SAPAILanguageModelV2("gpt-4o", settings, defaultConfig);

      expect(model).toBeDefined();
      expect(model.modelId).toBe("gpt-4o");
    });
  });

  describe("doGenerate", () => {
    it("should delegate to V3 model and transform response to V2 format", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      // Mock the V3 model's doGenerate method
      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [
          {
            text: "Hello, how can I help you?",
            type: "text",
          },
        ],
        finishReason: {
          raw: "stop",
          unified: "stop",
        },
        providerMetadata: {
          "sap-ai": { modelId: "gpt-4o" },
        },
        request: {
          body: { messages: [] },
        },
        response: {
          body: { choices: [] },
          headers: { "x-request-id": "123" },
          id: "resp-123",
          modelId: "gpt-4o",
          timestamp: new Date(),
        },
        usage: {
          inputTokens: {
            total: 10,
          },
          outputTokens: {
            total: 20,
          },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Hello", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(mockDoGenerate).toHaveBeenCalled();
      expect(result.content).toEqual([
        {
          text: "Hello, how can I help you?",
          type: "text",
        },
      ]);
      expect(result.finishReason).toBe("stop");
      expect(result.usage).toEqual({
        inputTokens: 10,
        outputTokens: 20,
        totalTokens: 30,
      });
      expect(result.warnings).toEqual([]);
      expect(result.providerMetadata).toBeDefined();
      expect(result.response?.id).toBe("resp-123");
      expect(result.response?.headers).toBeDefined();
    });

    it("should convert finish reason from V3 object to V2 string", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: {
          raw: "length",
          unified: "length",
        },
        usage: {
          inputTokens: { total: 5 },
          outputTokens: { total: 10 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.finishReason).toBe("length");
      expect(typeof result.finishReason).toBe("string");
    });

    it("should convert usage from V3 nested format to V2 flat format", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        usage: {
          inputTokens: {
            cacheRead: 5,
            total: 100,
          },
          outputTokens: {
            reasoning: 10,
            total: 50,
          },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.usage).toEqual({
        cachedInputTokens: 5,
        inputTokens: 100,
        outputTokens: 50,
        reasoningTokens: 10,
        totalTokens: 150,
      });
    });

    it("should convert warnings from V3 to V2 format", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        usage: {
          inputTokens: { total: 10 },
          outputTokens: { total: 20 },
        },
        warnings: [
          {
            details: "Tool not supported",
            feature: "custom-tool",
            type: "unsupported" as const,
          },
        ],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.warnings).toHaveLength(1);
      expect(result.warnings[0]).toEqual({
        message: "Unsupported feature: custom-tool. Tool not supported",
        type: "other",
      });
    });

    it("should handle tool calls in content", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [
          {
            input: { query: "weather" },
            toolCallId: "call-123",
            toolName: "get_weather",
            type: "tool-call",
          },
        ],
        finishReason: { raw: "tool-calls", unified: "tool-calls" },
        usage: {
          inputTokens: { total: 10 },
          outputTokens: { total: 5 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "What's the weather?", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.content).toEqual([
        {
          input: { query: "weather" },
          toolCallId: "call-123",
          toolName: "get_weather",
          type: "tool-call",
        },
      ]);
      expect(result.finishReason).toBe("tool-calls");
    });

    it("should pass through request and response metadata", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockTimestamp = new Date("2024-01-01T00:00:00Z");
      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        request: {
          body: { messages: [{ content: "test", role: "user" }] },
        },
        response: {
          body: { choices: [] },
          headers: { "content-type": "application/json" },
          id: "chatcmpl-123",
          modelId: "gpt-4o",
          timestamp: mockTimestamp,
        },
        usage: {
          inputTokens: { total: 10 },
          outputTokens: { total: 20 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.request?.body).toEqual({
        messages: [{ content: "test", role: "user" }],
      });
      expect(result.response?.id).toBe("chatcmpl-123");
      expect(result.response?.modelId).toBe("gpt-4o");
      expect(result.response?.timestamp).toBe(mockTimestamp);
      expect(result.response?.headers).toEqual({ "content-type": "application/json" });
    });

    it("should handle missing optional fields gracefully", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        usage: {
          inputTokens: { total: 10 },
          outputTokens: { total: 20 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.providerMetadata).toBeUndefined();
      expect(result.request).toBeUndefined();
      expect(result.response).toBeUndefined();
    });
  });

  describe("doStream", () => {
    it("should delegate to V3 model and transform stream to V2 format", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      // Create a mock V3 stream
      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            type: "response-metadata",
            value: {
              id: "resp-123",
              modelId: "gpt-4o",
              timestamp: new Date(),
            },
          });
          controller.enqueue({
            delta: "Hello",
            id: "text-0",
            type: "text-delta",
          });
          controller.enqueue({
            delta: " world",
            id: "text-0",
            type: "text-delta",
          });
          controller.enqueue({
            finishReason: { raw: "stop", unified: "stop" },
            type: "finish",
            usage: {
              inputTokens: { total: 5 },
              outputTokens: { total: 10 },
            },
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        request: {
          body: { messages: [] },
        },
        response: {
          headers: { "x-request-id": "123" },
        },
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Hello", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(mockDoStream).toHaveBeenCalled();
      expect(result.stream).toBeDefined();
      expect(result.request?.body).toEqual({ messages: [] });
      expect(result.response?.headers).toEqual({ "x-request-id": "123" });

      // Read the stream to verify V2 format
      const reader = result.stream.getReader();
      const parts = [];
      let done = false;

      while (!done) {
        const { done: streamDone, value } = await reader.read();
        done = streamDone;
        if (value) {
          parts.push(value);
        }
      }

      // Verify stream parts are in V2 format
      expect(parts.length).toBeGreaterThan(0);
      const textDelta = parts.find((p) => p.type === "text-delta");
      expect(textDelta).toBeDefined();

      const finish = parts.find((p) => p.type === "finish");
      expect(finish).toBeDefined();
      if (finish?.type === "finish") {
        // V2 finish reason should be a string
        expect(typeof finish.finishReason).toBe("string");
        // V2 usage should be flat format
        expect(finish.usage).toHaveProperty("totalTokens");
      }
    });

    it("should convert V3 stream finish event to V2 format", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            finishReason: { raw: "length", unified: "length" },
            type: "finish",
            usage: {
              inputTokens: { cacheRead: 5, total: 100 },
              outputTokens: { reasoning: 10, total: 50 },
            },
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      const reader = result.stream.getReader();
      const parts = [];
      let done = false;

      while (!done) {
        const { done: streamDone, value } = await reader.read();
        done = streamDone;
        if (value) {
          parts.push(value);
        }
      }

      const finish = parts.find((p) => p.type === "finish");
      expect(finish).toBeDefined();
      if (finish?.type === "finish") {
        expect(finish.finishReason).toBe("length");
        expect(finish.usage).toEqual({
          cachedInputTokens: 5,
          inputTokens: 100,
          outputTokens: 50,
          reasoningTokens: 10,
          totalTokens: 150,
        });
      }
    });

    it("should handle text-delta events in stream", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            delta: "Hello",
            id: "text-0",
            type: "text-delta",
          });
          controller.enqueue({
            delta: " world",
            id: "text-0",
            type: "text-delta",
          });
          controller.enqueue({
            finishReason: { raw: "stop", unified: "stop" },
            type: "finish",
            usage: {
              inputTokens: { total: 5 },
              outputTokens: { total: 2 },
            },
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Say hello", type: "text" }],
            role: "user",
          },
        ],
      });

      const reader = result.stream.getReader();
      const parts = [];
      let done = false;

      while (!done) {
        const { done: streamDone, value } = await reader.read();
        done = streamDone;
        if (value) {
          parts.push(value);
        }
      }

      const textDeltas = parts.filter((p) => p.type === "text-delta");
      expect(textDeltas).toHaveLength(2);
      expect(textDeltas[0]).toMatchObject({
        delta: "Hello",
        id: "text-0",
        type: "text-delta",
      });
      expect(textDeltas[1]).toMatchObject({
        delta: " world",
        id: "text-0",
        type: "text-delta",
      });
    });

    it("should handle tool calls in stream", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            input: { location: "Paris" },
            toolCallId: "call-123",
            toolName: "get_weather",
            type: "tool-call",
          });
          controller.enqueue({
            finishReason: { raw: "tool-calls", unified: "tool-calls" },
            type: "finish",
            usage: {
              inputTokens: { total: 10 },
              outputTokens: { total: 5 },
            },
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Weather in Paris?", type: "text" }],
            role: "user",
          },
        ],
      });

      const reader = result.stream.getReader();
      const parts = [];
      let done = false;

      while (!done) {
        const { done: streamDone, value } = await reader.read();
        done = streamDone;
        if (value) {
          parts.push(value);
        }
      }

      const toolCall = parts.find((p) => p.type === "tool-call");
      expect(toolCall).toBeDefined();
      if (toolCall?.type === "tool-call") {
        expect(toolCall.toolName).toBe("get_weather");
        expect(toolCall.toolCallId).toBe("call-123");
        expect(toolCall.input).toEqual({ location: "Paris" });
      }
    });

    it("should handle errors in stream", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockError = new Error("Stream error");
      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            error: mockError,
            type: "error",
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      const reader = result.stream.getReader();
      const parts = [];
      let done = false;

      while (!done) {
        const { done: streamDone, value } = await reader.read();
        done = streamDone;
        if (value) {
          parts.push(value);
        }
      }

      const error = parts.find((p) => p.type === "error");
      expect(error).toBeDefined();
      if (error?.type === "error") {
        expect(error.error).toBe(mockError);
      }
    });

    it("should pass through request and response metadata", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockV3Stream = new ReadableStream({
        start(controller) {
          controller.enqueue({
            finishReason: { raw: "stop", unified: "stop" },
            type: "finish",
            usage: {
              inputTokens: { total: 1 },
              outputTokens: { total: 1 },
            },
          });
          controller.close();
        },
      });

      const mockDoStream = vi.fn().mockResolvedValue({
        request: {
          body: { messages: [{ content: "test", role: "user" }] },
        },
        response: {
          headers: { "content-type": "text/event-stream", "x-request-id": "req-456" },
        },
        stream: mockV3Stream,
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doStream = mockDoStream;

      const result = await model.doStream({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.request?.body).toEqual({
        messages: [{ content: "test", role: "user" }],
      });
      expect(result.response?.headers).toEqual({
        "content-type": "text/event-stream",
        "x-request-id": "req-456",
      });
    });
  });

  describe("Edge Cases", () => {
    it("should handle empty warnings array", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        usage: {
          inputTokens: { total: 5 },
          outputTokens: { total: 10 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.warnings).toEqual([]);
    });

    it("should handle provider metadata correctly", async () => {
      const model = new SAPAILanguageModelV2("gpt-4o", {}, defaultConfig);

      const mockDoGenerate = vi.fn().mockResolvedValue({
        content: [{ text: "Test", type: "text" }],
        finishReason: { raw: "stop", unified: "stop" },
        providerMetadata: {
          "sap-ai": {
            customField: "value",
            modelId: "gpt-4o",
          },
        },
        usage: {
          inputTokens: { total: 5 },
          outputTokens: { total: 10 },
        },
        warnings: [],
      });

      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
      (model as any).v3Model.doGenerate = mockDoGenerate;

      const result = await model.doGenerate({
        prompt: [
          {
            content: [{ text: "Test", type: "text" }],
            role: "user",
          },
        ],
      });

      expect(result.providerMetadata).toEqual({
        "sap-ai": {
          customField: "value",
          modelId: "gpt-4o",
        },
      });
    });

    it("should handle different model IDs", () => {
      const models = [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
        "gemini-2.0-flash",
        "anthropic--claude-3-5-sonnet",
      ];

      for (const modelId of models) {
        const model = new SAPAILanguageModelV2(modelId, {}, defaultConfig);
        expect(model.modelId).toBe(modelId);
        expect(model.specificationVersion).toBe("v2");
      }
    });
  });
});
