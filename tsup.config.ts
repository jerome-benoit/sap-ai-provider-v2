import { defineConfig } from "tsup";

export default defineConfig([
  {
    clean: true,
    define: {
      __PACKAGE_VERSION__: JSON.stringify(
        (await import("./package.json", { with: { type: "json" } })).default.version,
      ),
    },
    dts: true,
    entry: ["src/index.ts"],
    format: ["cjs", "esm"],
    sourcemap: true,
  },
]);
