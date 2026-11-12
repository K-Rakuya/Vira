import { invoke as tauriInvoke } from '@tauri-apps/api/core';
import { listen as tauriListen, type UnlistenFn } from '@tauri-apps/api/event';

export function isTauriEnvironment(): boolean {
  return typeof window !== 'undefined' && ('__TAURI_INTERNALS__' in window || '__TAURI__' in window);
}

export type EventCallback<T = any> = (payload: T) => void;

class MockIPC {
  private listeners = new Map<string, Set<EventCallback>>();
  private mockTickInterval: any = null;
  private currentMode = 'vietnamese';

  public listen<T>(event: string, callback: EventCallback<T>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    const set = this.listeners.get(event)!;
    set.add(callback as EventCallback);
    return () => {
      set.delete(callback as EventCallback);
    };
  }

  public emit(event: string, payload: any): void {
    const set = this.listeners.get(event);
    if (set) {
      for (const cb of set) {
        try {
          cb(payload);
        } catch (err) {
          console.error(`Error in mock listener for ${event}:`, err);
        }
      }
    }
  }

  public async invoke<T = any>(command: string, args: Record<string, any> = {}): Promise<T> {
    console.log(`[MockIPC] invoke: ${command}`, args);

    switch (command) {
      case 'send_message': {
        const text = args.payload?.text || '';
        this.emit('subtitle_update', {
          text,
          state: 'Thinking',
          timestamp: Date.now(),
        });

        setTimeout(() => {
          const reply = `Xin chào! Tôi nhận được tin: "${text}". Hệ thống đang hoạt động ở chế độ Mock IPC Web dev!`;
          this.emit('subtitle_update', {
            text: reply,
            state: 'Speaking',
            timestamp: Date.now(),
          });
          this.startMockLipSync();
        }, 500);

        return {
          message_id: 'mock-' + Math.random().toString(36).slice(2, 9),
          status: 'completed',
        } as T;
      }

      case 'trigger_gesture':
      case 'play_gesture': {
        this.emit('gesture_event', {
          name: args.name,
          duration: args.duration || 2.0,
        });
        return undefined as T;
      }

      case 'set_emotion': {
        this.emit('emotion_event', {
          name: args.name,
          weight: args.weight ?? 1.0,
        });
        return undefined as T;
      }

      case 'set_tutor_mode': {
        this.currentMode = args.mode;
        return args.mode as T;
      }

      case 'toggle_mic': {
        const active = Boolean(args.active);
        this.emit('subtitle_update', {
          text: active ? 'Đang lắng nghe giọng nói của bạn…' : '',
          state: active ? 'Listening' : 'Idle',
          timestamp: Date.now(),
        });
        return active as T;
      }

      case 'save_settings': {
        return undefined as T;
      }

      case 'get_app_state': {
        return {
          state: 'idle',
          conversation_state: 'Idle',
          tutor_mode: this.currentMode,
          emotion: 'neutral',
          emotion_weight: 1.0,
          mic_active: false,
          is_speaking: false,
          audio_playing: false,
          current_subtitle: '',
          sample_clock: 0,
          current_time_ms: 0,
          history_length: 0,
        } as T;
      }

      default:
        console.warn(`[MockIPC] Unhandled command: ${command}`);
        return {} as T;
    }
  }

  private startMockLipSync(): void {
    if (this.mockTickInterval) clearInterval(this.mockTickInterval);
    let count = 0;
    const maxTicks = 120; // ~2 seconds of 60Hz ticks
    const visemes = ['aa', 'ih', 'ou', 'ee', 'oh'];

    this.mockTickInterval = setInterval(() => {
      count++;
      if (count > maxTicks) {
        clearInterval(this.mockTickInterval);
        this.mockTickInterval = null;
        this.emit('viseme_tick', {
          weights: { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 },
          sample_clock: count * 16,
          timestamp_ms: count * 16,
          is_speaking: false,
        });
        this.emit('subtitle_update', {
          text: 'Xin chào! Tôi là Vira, trợ lý ảo 3D đồng hành cùng bạn. Bạn cần tôi hỗ trợ gì hôm nay?',
          state: 'Idle',
          timestamp: Date.now(),
        });
        return;
      }

      const activeIdx = Math.floor((count / 12) % visemes.length);
      const intensity = Math.sin((count / 8) * Math.PI) * 0.5 + 0.5;
      const weights: Record<string, number> = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
      weights[visemes[activeIdx]] = intensity;

      this.emit('viseme_tick', {
        weights,
        sample_clock: count * 16,
        timestamp_ms: count * 16,
        is_speaking: true,
      });
    }, 16);
  }
}

const mockIPC = new MockIPC();

export class TauriBridge {
  public static isTauri(): boolean {
    return isTauriEnvironment();
  }

  public static async invoke<T = any>(cmd: string, args?: Record<string, any>): Promise<T> {
    if (this.isTauri()) {
      return await tauriInvoke<T>(cmd, args);
    }
    return mockIPC.invoke<T>(cmd, args);
  }

  public static async listen<T = any>(event: string, callback: EventCallback<T>): Promise<() => void> {
    if (this.isTauri()) {
      const unlisten: UnlistenFn = await tauriListen<T>(event, (ev) => callback(ev.payload));
      return () => unlisten();
    }
    return mockIPC.listen<T>(event, callback);
  }

  public static mockEmit(event: string, payload: any): void {
    mockIPC.emit(event, payload);
  }
}
