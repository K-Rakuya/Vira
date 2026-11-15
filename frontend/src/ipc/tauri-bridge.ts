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
  private memoryStorage = new Map<string, string>();

  private getItem(key: string): string | null {
    if (typeof localStorage !== 'undefined') {
      try {
        return localStorage.getItem(key);
      } catch {
        // ignore
      }
    }
    return this.memoryStorage.get(key) || null;
  }

  private setItem(key: string, value: string): void {
    if (typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(key, value);
        return;
      } catch {
        // ignore
      }
    }
    this.memoryStorage.set(key, value);
  }

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

      case 'ingest_document': {
        const payload = args.payload || {};
        const doc = {
          id: 'doc_' + Date.now(),
          title: payload.title || 'Untitled',
          content: payload.content || '',
          chunk_count: Math.max(1, Math.ceil((payload.content?.length || 0) / 300)),
          created_at: new Date().toISOString(),
        };
        const current = JSON.parse(this.getItem('vira_mock_docs') || '[]');
        current.push(doc);
        this.setItem('vira_mock_docs', JSON.stringify(current));
        return doc as T;
      }

      case 'list_documents': {
        const current = JSON.parse(this.getItem('vira_mock_docs') || '[]');
        return current as T;
      }

      case 'delete_document': {
        const id = args.doc_id || args.id;
        const current = JSON.parse(this.getItem('vira_mock_docs') || '[]');
        const filtered = current.filter((d: any) => d.id !== id);
        this.setItem('vira_mock_docs', JSON.stringify(filtered));
        return undefined as T;
      }

      case 'search_knowledge': {
        const current = JSON.parse(this.getItem('vira_mock_docs') || '[]');
        const q = String(args.query || '').toLowerCase();
        const hits = current
          .filter((d: any) => d.title.toLowerCase().includes(q) || d.content.toLowerCase().includes(q))
          .map((d: any) => ({
            chunk_id: d.id + '_c0',
            document_id: d.id,
            title: d.title,
            text: d.content.slice(0, 150),
            score: 0.85,
            citation: `[${d.title}]`,
          }));
        return hits as T;
      }

      case 'create_rich_flashcard':
      case 'create_flashcard': {
        const payload = args.payload || {};
        const card = {
          id: 'fc_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6),
          front: payload.front,
          back: payload.back,
          phonetic: payload.phonetic || '/fəˈnɛtɪk/',
          context_sentence: payload.context_sentence || '',
          context_translation: payload.context_translation || '',
          part_of_speech: payload.part_of_speech || 'phrase',
          cefr_level: payload.cefr_level || 'A1',
          tags: Array.isArray(payload.tags) ? payload.tags : ['conversation'],
          state: 0,
          stability: 2.0,
          difficulty: 5.0,
          due: Date.now(),
          reps: 0,
          lapses: 0,
        };
        const cards = JSON.parse(this.getItem('vira_mock_cards') || '[]');
        cards.push(card);
        this.setItem('vira_mock_cards', JSON.stringify(cards));
        return card as T;
      }

      case 'generate_flashcard_ai_suggestion': {
        const text = String(args.text || args.payload?.text || 'Xin chào, tôi là Vira').trim();
        const words = text.split(/\s+/).filter(Boolean);
        const front = words[0] || 'Vira';
        return {
          front,
          back: 'Từ vựng quan trọng được trích xuất từ cuộc trò chuyện',
          phonetic: '/ˈvaɪrə/',
          context_sentence: text,
          context_translation: 'Bản dịch ngữ cảnh tự động của Vira.',
          part_of_speech: 'phrase',
          cefr_level: 'B1',
          tags: ['auto-extracted', 'chat'],
        } as T;
      }

      case 'list_flashcards':
      case 'get_due_flashcards': {
        const cards = JSON.parse(this.getItem('vira_mock_cards') || '[]');
        return cards as T;
      }

      case 'review_flashcard': {
        const cardId = args.card_id || args.payload?.card_id;
        const cards = JSON.parse(this.getItem('vira_mock_cards') || '[]');
        const card = cards.find((c: any) => c.id === cardId);
        if (card) {
          card.reps = (card.reps || 0) + 1;
          card.stability = (card.stability || 2.0) * 1.5;
          this.setItem('vira_mock_cards', JSON.stringify(cards));
          return card as T;
        }
        return {} as T;
      }

      case 'delete_flashcard': {
        const cardId = args.card_id;
        const cards = JSON.parse(this.getItem('vira_mock_cards') || '[]');
        const filtered = cards.filter((c: any) => c.id !== cardId);
        this.setItem('vira_mock_cards', JSON.stringify(filtered));
        return undefined as T;
      }

      case 'preview_flashcard_intervals': {
        return {
          again: '< 10m',
          hard: '1 ngày',
          good: '3 ngày',
          easy: '7 ngày',
        } as T;
      }

      case 'get_pronunciation_reference_audio': {
        const target = String(args.target_text || args.payload?.target_text || 'Xin chào');
        return {
          target_text: target,
          audio_base64: 'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=',
          sample_rate: 24000,
          status: 'ok',
        } as T;
      }

      case 'replay_user_recording': {
        return {
          audio_base64: 'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=',
          sample_rate: 16000,
          status: 'ok',
        } as T;
      }

      case 'assess_pronunciation_stream':
      case 'assess_pronunciation': {
        const target = args.payload?.target_text || args.target_text || 'Xin chào';
        const rawWords = target.split(/\s+/).filter(Boolean);
        const words = rawWords.map((w: string, idx: number) => {
          const accScore = 88 + (idx % 3) * 4;
          return {
            word: w,
            accuracy_score: accScore,
            error_type: accScore >= 80 ? 'None' : 'Mispronounced',
            phonemes: [
              {
                phoneme: w.slice(0, 1) || 'a',
                score: accScore + 2,
                accuracy: 'correct',
                viseme_hint: { aa: 0.8, ih: 0.1, ou: 0, ee: 0.1, oh: 0 },
                recommended_viseme: 'aa',
              },
              {
                phoneme: w.slice(1) || 'h',
                score: accScore - 2,
                accuracy: accScore >= 80 ? 'correct' : 'minorflaw',
                viseme_hint: { aa: 0.1, ih: 0.7, ou: 0, ee: 0.2, oh: 0 },
                recommended_viseme: 'ih',
              },
            ],
          };
        });

        const phonemeScores = words.flatMap((w: any, wIdx: number) =>
          w.phonemes.map((p: any, pIdx: number) => ({
            phoneme: p.phoneme,
            score: p.score,
            accuracy: p.accuracy,
            start_ms: (wIdx * 2 + pIdx) * 150,
            end_ms: (wIdx * 2 + pIdx + 1) * 150,
            viseme_hint: p.viseme_hint,
            recommended_viseme: p.recommended_viseme,
          }))
        );

        return {
          overall_score: 91.5,
          cefr_level: 'C1',
          accuracy_score: 92.0,
          fluency_score: 89.0,
          completeness_score: 95.0,
          prosody_score: 88.0,
          words,
          phoneme_scores: phonemeScores,
          mispronounced_count: 0,
          total_phonemes: phonemeScores.length,
          feedback_tips: [
            'Ngữ điệu và trọng âm câu tự nhiên, nhịp điệu nói chuẩn người bản xứ.',
            'Mở rộng khẩu hình vòm họng hơn một chút ở các nguyên âm để tăng độ dày âm.',
          ],
        } as T;
      }

      case 'execute_tool': {
        return {
          tool_call_id: 'tool_' + Date.now(),
          name: args.payload?.tool_name || 'builtin',
          content: 'Mock execution result',
          is_error: false,
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
