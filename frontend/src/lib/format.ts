/** Formatting helpers shared by the evaluation and runs pages. */

export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

/** Format a 95% confidence interval as "71.1–86.7%". */
export const formatCi = (low: number, high: number) =>
  `${(low * 100).toFixed(1)}–${(high * 100).toFixed(1)}%`;
