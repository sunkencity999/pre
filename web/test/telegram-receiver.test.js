// Tests for src/telegram-receiver.js — background polling
jest.mock('../src/constants');

const receiver = require('../src/telegram-receiver');

describe('telegram-receiver', () => {
  afterAll(() => receiver.shutdown());

  test('getStatus returns inactive before init', () => {
    const s = receiver.getStatus();
    expect(s.active).toBe(false);
    expect(s.username).toBeNull();
    expect(s.lastOffset).toBe(0);
  });

  test('isActive returns false before init', () => {
    expect(receiver.isActive()).toBe(false);
  });

  test('init returns inactive without bot token', async () => {
    const result = await receiver.init(jest.fn());
    expect(result.active).toBe(false);
  });

  test('shutdown is safe when not active', () => {
    expect(() => receiver.shutdown()).not.toThrow();
    expect(receiver.isActive()).toBe(false);
  });
});
