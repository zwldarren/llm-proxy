import { describe, it, expect } from "vitest";
import { openaiAdapter } from "./openaiAdapter";

interface TestTool {
  name: string;
  description: string;
  parameters: string;
  enabled: boolean;
}

describe("openaiAdapter formatMessages", () => {
  it("filters out tool_calls entries with missing id", () => {
    const result = openaiAdapter.formatMessages([
      {
        role: "assistant",
        content: "Test",
        tool_calls: [
          // @ts-expect-error — testing null id filter
          { id: null, type: "function", function: { name: "f1", arguments: "{}" } },
          { id: "valid", type: "function", function: { name: "f2", arguments: "{}" } },
        ],
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const msg = result[0] as any;
    const tcs = msg.tool_calls;
    expect(tcs).toBeDefined();
    expect(tcs).toHaveLength(1);
    expect(tcs[0].id).toBe("valid");
  });
});

describe("openaiAdapter formatTools", () => {
  it("parses JSON parameters properly", () => {
    const result = openaiAdapter.formatTools([
      {
        name: "test",
        description: "Test tool",
        parameters: '{"type":"object","properties":{"x":{"type":"string"}}}',
        enabled: true,
      },
    ] as TestTool[]) as { function: { parameters: Record<string, unknown> } }[];
    expect(result).toHaveLength(1);
    expect(result[0]!.function.parameters).toEqual({
      type: "object",
      properties: { x: { type: "string" } },
    });
  });

  it("handles empty string parameters", () => {
    const result = openaiAdapter.formatTools([
      { name: "test", description: "", parameters: "", enabled: true },
    ] as TestTool[]) as {
      function: { parameters: Record<string, unknown>; description?: string };
    }[];
    expect(result).toHaveLength(1);
    expect(result[0]!.function.parameters).toEqual({});
    expect(result[0]!.function.description).toBeUndefined();
  });
});
