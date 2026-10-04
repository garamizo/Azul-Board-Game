import { describe, expect, it } from 'vitest';
import { oneAtATime } from './submit';

describe('oneAtATime', () => {
  it('ignores calls while one is in flight', async () => {
    let calls = 0;
    let release!: () => void;
    const run = oneAtATime(() => { calls++; return new Promise<void>((r) => { release = r; }); });
    const first = run();
    const second = run();
    expect(calls).toBe(1);
    expect(await second).toBeUndefined();
    release();
    await first;
    run();
    expect(calls).toBe(2);
  });

  it('frees itself after a failure', async () => {
    let calls = 0;
    const run = oneAtATime(async () => { calls++; throw new Error('x'); });
    await expect(run()).rejects.toThrow('x');
    await expect(run()).rejects.toThrow('x');
    expect(calls).toBe(2);
  });
});
