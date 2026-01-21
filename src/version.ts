/**
 * Package version injected at build time via tsup define.
 * @module
 */

declare const __PACKAGE_VERSION__: string | undefined;

/** Current package version from package.json, injected at build time. */
export const VERSION: string =
  typeof __PACKAGE_VERSION__ !== "undefined" ? __PACKAGE_VERSION__ : "0.0.0-test";
