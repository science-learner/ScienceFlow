const DEBUG_KEY = "scienceflow.debug";

export const debug = {
  get enabled(): boolean {
    return localStorage.getItem(DEBUG_KEY) === "1";
  },
  log(prefix: string, ...args: unknown[]): void {
    if (this.enabled) console.log(`[${prefix}]`, ...args);
  },
  warn(prefix: string, ...args: unknown[]): void {
    if (this.enabled) console.warn(`[${prefix}]`, ...args);
  },
  error(prefix: string, ...args: unknown[]): void {
    if (this.enabled) console.error(`[${prefix}]`, ...args);
  },
};
