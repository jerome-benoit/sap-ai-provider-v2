/**
 * Deep merge utility with prototype pollution protection.
 * @module deep-merge
 */

/** Keys blocked to prevent prototype pollution. */
const DANGEROUS_KEYS = new Set(["__proto__", "constructor", "prototype"]);

/** Maximum recursion depth to prevent stack overflow. */
const MAX_DEPTH = 100;

/**
 * Deep merges multiple objects recursively.
 *
 * ## Merge Behavior
 * - **Plain objects**: Merged recursively (properties combined)
 * - **Arrays**: Replaced (last array wins, no concatenation)
 * - **Primitives**: Replaced (last value wins)
 * - **Special objects** (Date, RegExp, Map, Set, etc.): Replaced (not merged)
 * - **null/undefined**: Treated as values (can override)
 *
 * ## Security
 * - Protected against prototype pollution (`__proto__`, `constructor`, `prototype` are blocked)
 * - Circular references are detected and throw an error
 *
 * ## Performance
 * - Time complexity: O(n) where n = total number of properties across all levels
 * - Space complexity: O(d) where d = maximum nesting depth
 * @param sources - Objects to merge (later objects override earlier ones)
 * @returns New merged object (source objects are not mutated)
 * @throws {Error} If circular reference is detected
 * @throws {Error} If maximum recursion depth is exceeded
 * @example
 * ```typescript
 * // Basic deep merge
 * const result = deepMerge(
 *   { a: { b: 1, c: 2 } },
 *   { a: { c: 3, d: 4 } }
 * );
 * // Result: { a: { b: 1, c: 3, d: 4 } }
 * ```
 * @example
 * ```typescript
 * // Arrays are replaced, not merged
 * const result = deepMerge(
 *   { arr: [1, 2, 3] },
 *   { arr: [4, 5] }
 * );
 * // Result: { arr: [4, 5] }
 * ```
 * @example
 * ```typescript
 * // Multiple objects
 * const result = deepMerge(
 *   { a: 1 },
 *   { b: 2 },
 *   { c: 3 }
 * );
 * // Result: { a: 1, b: 2, c: 3 }
 * ```
 */
export function deepMerge<T extends Record<string, unknown>>(
  ...sources: (Partial<T> | undefined)[]
): T {
  return mergeInternal(sources as Record<string, unknown>[]) as T;
}

/**
 * Deep merges two objects with better type inference for common cases.
 *
 * This is a convenience wrapper around deepMerge for the common two-object case.
 * @param target - Base object.
 * @param source - Object to merge into target.
 * @returns New merged object.
 * @example
 * ```typescript
 * const defaults = { timeout: 1000, retry: { max: 3, delay: 100 } };
 * const overrides = { retry: { max: 5 } };
 * const config = deepMergeTwo(defaults, overrides);
 * // Result: { timeout: 1000, retry: { max: 5, delay: 100 } }
 * ```
 */
export function deepMergeTwo<T extends Record<string, unknown>>(
  target: T | undefined,
  source: Partial<T> | undefined,
): T {
  return deepMerge(target, source);
}

/**
 * Deep clones a plain object with circular reference detection.
 * @param obj - Object to clone.
 * @param seen - Set of visited objects for cycle detection.
 * @param depth - Current recursion depth.
 * @returns Cloned object.
 * @internal
 */
function cloneDeep(
  obj: Record<string, unknown>,
  seen: WeakSet<object>,
  depth: number,
): Record<string, unknown> {
  if (depth > MAX_DEPTH) {
    throw new Error("Maximum merge depth exceeded");
  }
  if (seen.has(obj)) {
    throw new Error("Circular reference detected during deep merge");
  }
  seen.add(obj);

  const result: Record<string, unknown> = {};
  for (const key of Object.keys(obj)) {
    if (!isSafeKey(key)) continue;
    const value = obj[key];
    result[key] = isPlainObject(value) ? cloneDeep(value, seen, depth + 1) : value;
  }
  return result;
}

/**
 * Type guard for plain objects (excludes arrays, Date, RegExp, class instances, etc.).
 * @param value - Value to check.
 * @returns True if value is a plain object.
 * @internal
 */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== "object") return false;
  const proto = Object.getPrototypeOf(value) as unknown;
  return proto === Object.prototype || proto === null;
}

/**
 * Checks if a key is safe to merge (not a prototype pollution vector).
 * @param key - Object key to check.
 * @returns True if key is safe to merge.
 * @internal
 */
function isSafeKey(key: string): boolean {
  return !DANGEROUS_KEYS.has(key);
}

/**
 * Merges multiple source objects into a single result.
 * @param sources - Array of objects to merge.
 * @returns Merged result object.
 * @internal
 */
function mergeInternal(sources: (Record<string, unknown> | undefined)[]): Record<string, unknown> {
  let result: Record<string, unknown> = {};
  for (const source of sources) {
    if (source == null) continue;
    result = mergeTwo(result, source, new WeakSet(), 0);
  }
  return result;
}

/**
 * Recursively merges two objects with depth tracking and circular reference detection.
 * @param target - Target object to merge into.
 * @param source - Source object to merge from.
 * @param seen - Set of visited objects for cycle detection.
 * @param depth - Current recursion depth.
 * @returns Merged result object.
 * @internal
 */
function mergeTwo(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
  seen: WeakSet<object>,
  depth: number,
): Record<string, unknown> {
  if (depth > MAX_DEPTH) {
    throw new Error("Maximum merge depth exceeded");
  }
  if (seen.has(source)) {
    throw new Error("Circular reference detected during deep merge");
  }
  seen.add(source);

  for (const key of Object.keys(source)) {
    if (!isSafeKey(key)) continue;

    const sourceValue = source[key];
    const targetValue = target[key];

    if (isPlainObject(sourceValue) && isPlainObject(targetValue)) {
      target[key] = mergeTwo({ ...targetValue }, sourceValue, seen, depth + 1);
    } else if (isPlainObject(sourceValue)) {
      target[key] = cloneDeep(sourceValue, seen, depth + 1);
    } else {
      target[key] = sourceValue;
    }
  }
  return target;
}
