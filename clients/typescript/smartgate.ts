// Minimal TypeScript/JavaScript client for a smartgate server (Node 18+ / browsers / Deno / Bun).
export interface GenerateOptions {
  prompt?: string;
  messages?: { role: "user" | "assistant"; content: string }[];
  system?: string;
  max_tokens?: number;
  tier?: "small" | "medium" | "large";
  model?: string;
  use_cache?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface GatewayResponse {
  text: string;
  model: string;
  provider: string;
  usage: { input_tokens: number; output_tokens: number };
  cost_usd: number;
  cache_hit: boolean;
  cache_similarity: number | null;
  saved_usd: number;
  route_reason: string;
  latency_ms: number;
  request_id: string;
}

export class SmartGateError extends Error {
  constructor(public status: number, public payload: unknown) {
    super(`smartgate HTTP ${status}: ${JSON.stringify(payload)}`);
  }
}

export class SmartGateClient {
  constructor(private baseUrl = "http://127.0.0.1:8080", private apiKey?: string) {}

  private async call<T>(method: string, path: string, body?: unknown): Promise<T> {
    const res = await fetch(this.baseUrl.replace(/\/$/, "") + path, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new SmartGateError(res.status, data);
    return data as T;
  }

  generate(opts: GenerateOptions): Promise<GatewayResponse> {
    return this.call("POST", "/v1/generate", opts);
  }
  analyze(opts: GenerateOptions) {
    return this.call<Record<string, unknown>>("POST", "/v1/analyze", opts);
  }
  stats() {
    return this.call<Record<string, unknown>>("GET", "/v1/stats");
  }
  costs() {
    return this.call<Record<string, unknown>>("GET", "/v1/costs");
  }
}
