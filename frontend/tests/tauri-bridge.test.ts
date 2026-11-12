import { describe, expect, it } from 'bun:test';
import { TauriBridge } from '../src/ipc/tauri-bridge';

describe('TauriBridge & Mock IPC Fallback', () => {
  it('correctly reports standalone browser mode outside Tauri', () => {
    // When running under tests or node/bun without Tauri window internals
    expect(typeof TauriBridge.isTauri()).toBe('boolean');
  });

  it('invokes mock send_message and receives response', async () => {
    const res = await TauriBridge.invoke<{ message_id: string; status: string }>('send_message', {
      payload: { text: 'Xin chào Vira' },
    });
    expect(res).toBeDefined();
    expect(res.message_id).toBeDefined();
    expect(res.status).toBe('completed');
  });

  it('invokes set_tutor_mode and get_app_state in mock mode', async () => {
    const mode = await TauriBridge.invoke<string>('set_tutor_mode', { mode: 'japaneseTutor' });
    expect(mode).toBe('japaneseTutor');

    const state = await TauriBridge.invoke<any>('get_app_state');
    expect(state).toBeDefined();
    expect(state.tutor_mode).toBe('japaneseTutor');
  });

  it('subscribes and emits events via mock IPC event bus', async () => {
    let received: any = null;
    const unlisten = await TauriBridge.listen('mock_event', (payload) => {
      received = payload;
    });

    TauriBridge.mockEmit('mock_event', { hello: 'world' });
    expect(received).toEqual({ hello: 'world' });

    unlisten();
    TauriBridge.mockEmit('mock_event', { hello: 'after_unlisten' });
    expect(received).toEqual({ hello: 'world' });
  });

  it('supports toggle_mic in mock mode', async () => {
    const active = await TauriBridge.invoke<boolean>('toggle_mic', { active: true });
    expect(active).toBe(true);

    const inactive = await TauriBridge.invoke<boolean>('toggle_mic', { active: false });
    expect(inactive).toBe(false);
  });
});
