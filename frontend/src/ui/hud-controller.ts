import { VRMViewer } from '../stage/vrm-viewer.js';
import { TauriBridge } from '../ipc/tauri-bridge.js';
import { parseKinematicCues } from '../stage/procedural-animator.js';

export interface TutorPreset {
  id: string;
  name: string;
  language: string;
  systemPrompt: string;
}

export const TUTOR_PRESETS: Record<string, TutorPreset> = {
  vietnamese: {
    id: 'vietnamese',
    name: 'Vietnamese Tutor',
    language: 'vi-VN',
    systemPrompt:
      'Bạn là Vira, gia sư tiếng Việt thông thái, dịu dàng và nhiệt huyết. Trả lời thẳng vào trọng tâm câu hỏi của người dùng, súc tích, mạch lạc và tự nhiên. Không nói lan man vòng vo ngoài lề trước khi vào chủ đề chính. Diễn đạt hành động ngắn gọn nếu cần thiết. Bạn hướng dẫn người học từ phát âm, dấu thanh, từ vựng đến ngữ pháp phong phú của tiếng Việt.',
  },
  english: {
    id: 'english',
    name: 'English Tutor',
    language: 'en-US',
    systemPrompt:
      'You are Vira, an encouraging, patient English tutor. Answer directly to the core of user questions, concisely, coherently, and naturally. Do not ramble or digress before addressing the main topic. Keep action descriptions brief if necessary. Help the user master natural conversation, vocabulary, and grammar with gentle corrections.',
  },
  japanese: {
    id: 'japanese',
    name: 'Japanese Tutor',
    language: 'ja-JP',
    systemPrompt:
      'あなたはViraです。親切で丁寧な日本語の先生です。質問の核心に直接、簡潔かつ自然に答えてください。本題に入る前の無駄な前置きや遠回しな表現は避けてください。行動描写が必要な場合は簡潔に行ってください。学習者の日本語の会話や文法を優しくサポートしてください。',
  },
  general_assistant: {
    id: 'general_assistant',
    name: 'General Assistant',
    language: 'vi-VN',
    systemPrompt:
      'Bạn là Vira, trợ lý ảo 3D thông minh, chu đáo và thân thiện. Trả lời thẳng vào trọng tâm câu hỏi của người dùng, súc tích, mạch lạc và tự nhiên. Không nói lan man vòng vo ngoài lề trước khi vào chủ đề chính. Diễn đạt hành động ngắn gọn nếu cần thiết.',
  },
};

// Aliases for compatibility
TUTOR_PRESETS.vietnameseTutor = TUTOR_PRESETS.vietnamese;
TUTOR_PRESETS.englishTutor = TUTOR_PRESETS.english;
TUTOR_PRESETS.japaneseTutor = TUTOR_PRESETS.japanese;
TUTOR_PRESETS.general = TUTOR_PRESETS.general_assistant;

export function normalizeTutorMode(modeKey: string): string {
  const lower = String(modeKey || '').trim().toLowerCase().replace(/[- ]/g, '_');
  if (
    lower === 'vietnamese' ||
    lower === 'vietnamesetutor' ||
    lower === 'vietnamese_tutor' ||
    lower === 'vi' ||
    lower === 'vn'
  ) {
    return 'vietnamese';
  }
  if (
    lower === 'english' ||
    lower === 'englishtutor' ||
    lower === 'english_tutor' ||
    lower === 'en'
  ) {
    return 'english';
  }
  if (
    lower === 'japanese' ||
    lower === 'japanesetutor' ||
    lower === 'japanese_tutor' ||
    lower === 'ja' ||
    lower === 'jp'
  ) {
    return 'japanese';
  }
  return 'general_assistant';
}

const STATUS_LABELS: Record<string, string> = {
  idle: 'Ready',
  listening: 'Listening',
  thinking: 'Thinking',
  speaking: 'Speaking',
  error: 'Error',
};

const HOLD_MS = 320;

export class HudController {
  private vrmViewer: VRMViewer;
  private el: Record<string, any> = {};
  private currentModeKey = 'vietnamese';
  private toastTimer: any = null;
  private openDrawerName: string | null = null;
  private micDownAt = 0;
  private isMicActive = false;
  private spaceHeld = false;
  private unlisteners: Array<() => void> = [];
  private speechRec: any = null;
  private dueCards: Array<any> = [];
  private currentStudyCard: any = null;
  private speechQueue: SpeechSynthesisUtterance[] = [];
  private isSpeakingSpeech = false;

  constructor(vrmViewer: VRMViewer) {
    this.vrmViewer = vrmViewer;
  }

  public async init(): Promise<void> {
    this.cacheElements();
    this.bindDomEvents();
    this.initSpeechRecognition();
    await this.setupIpcListeners();
    await this.refreshDocumentList();
    await this.refreshFlashcards();
    this.setStatus('idle');

    // Sync initial state from backend
    try {
      const appState = await TauriBridge.invoke('get_app_state');
      if (appState?.tutor_mode) {
        this.setTutorModeUI(appState.tutor_mode);
      }
    } catch {
      // ignore
    }
  }

  private cacheElements(): void {
    const $ = (id: string) => document.getElementById(id);
    this.el = {
      chatInput: $('chat-input') as HTMLInputElement,
      btnSend: $('btn-send') as HTMLButtonElement,
      btnMic: $('btn-mic') as HTMLButtonElement,
      subtitleCard: $('subtitle-card') as HTMLElement,
      subtitleBadge: $('subtitle-badge') as HTMLElement,
      subtitleText: $('subtitle-text') as HTMLElement,
      thoughtCard: $('thought-card') as HTMLDetailsElement,
      thoughtBadge: $('thought-badge') as HTMLElement,
      thoughtText: $('thought-text') as HTMLElement,
      tutorSelect: $('tutor-select') as HTMLSelectElement,
      brandModeLabel: $('brand-mode-label') as HTMLElement,
      btnGestures: $('btn-gestures') as HTMLButtonElement,
      gesturePanel: $('gesture-panel') as HTMLElement,
      btnResetCamera: $('btn-reset-camera') as HTMLButtonElement,

      // Knowledge drawer
      btnKnowledge: $('btn-knowledge-toggle') as HTMLButtonElement,
      btnCloseKnowledge: $('btn-close-knowledge') as HTMLButtonElement,
      knowledgeDrawer: $('knowledge-drawer') as HTMLElement,
      docTitle: $('doc-title-input') as HTMLInputElement,
      docContent: $('doc-content-input') as HTMLTextAreaElement,
      docDrop: $('doc-drop-zone') as HTMLElement,
      docFile: $('doc-file-input') as HTMLInputElement,
      btnIngest: $('btn-ingest-doc') as HTMLButtonElement,
      docList: $('doc-list-container') as HTMLElement,

      // Settings drawer
      btnSettings: $('btn-settings-toggle') as HTMLButtonElement,
      btnCloseSettings: $('btn-close-settings') as HTMLButtonElement,
      settingsDrawer: $('settings-drawer') as HTMLElement,
      baseUrl: $('setting-base-url') as HTMLInputElement,
      model: $('setting-model') as HTMLInputElement,
      apiKey: $('setting-api-key') as HTMLInputElement,
      temperature: $('setting-temperature') as HTMLInputElement,
      outTemperature: $('out-temperature') as HTMLElement,
      vrmDrop: $('vrm-drop-zone') as HTMLElement,
      vrmFile: $('vrm-file-input') as HTMLInputElement,
      systemPrompt: $('setting-system-prompt') as HTMLTextAreaElement,
      voiceSelect: $('setting-voice-select') as HTMLSelectElement,
      rate: $('setting-speech-rate') as HTMLInputElement,
      outRate: $('out-rate') as HTMLElement,
      rag: $('setting-rag-enabled') as HTMLInputElement,
      btnSave: $('btn-save-settings') as HTMLButtonElement,

      // Flashcards drawer
      btnFlashcards: $('btn-flashcards-toggle') as HTMLButtonElement,
      dueBadge: $('due-badge') as HTMLElement,
      flashcardsDrawer: $('flashcards-drawer') as HTMLElement,
      btnCloseFlashcards: $('btn-close-flashcards') as HTMLButtonElement,
      fcDuePill: $('fc-due-pill') as HTMLElement,
      fcTotalCount: $('fc-total-count') as HTMLElement,
      fcActiveCard: $('fc-active-card') as HTMLElement,
      fcCardLevel: $('fc-card-level') as HTMLElement,
      fcCardFront: $('fc-card-front') as HTMLElement,
      fcCardBack: $('fc-card-back') as HTMLElement,
      btnFcReveal: $('btn-fc-reveal') as HTMLButtonElement,
      fcRatingActions: $('fc-rating-actions') as HTMLElement,
      fcInputFront: $('fc-input-front') as HTMLInputElement,
      fcInputBack: $('fc-input-back') as HTMLTextAreaElement,
      fcSelectLevel: $('fc-select-level') as HTMLSelectElement,
      btnFcSave: $('btn-fc-save') as HTMLButtonElement,
      fcListContainer: $('fc-list-container') as HTMLElement,

      // Pronunciation modal
      btnPronounce: $('btn-pronounce-toggle') as HTMLButtonElement,
      pronounceModal: $('pronounce-modal') as HTMLDialogElement,
      btnClosePronounce: $('btn-close-pronounce') as HTMLButtonElement,
      pronounceTarget: $('pronounce-target-input') as HTMLInputElement,
      btnPronounceAssess: $('btn-pronounce-assess') as HTMLButtonElement,
      pronounceResult: $('pronounce-result') as HTMLElement,
      pronounceOverallScore: $('pronounce-overall-score') as HTMLElement,
      pronounceAccuracyLabel: $('pronounce-accuracy-label') as HTMLElement,
      pronouncePhonemes: $('pronounce-phonemes-container') as HTMLElement,

      toast: $('toast') as HTMLElement,
      scrim: $('scrim') as HTMLElement,
    };
  }

  private async setupIpcListeners(): Promise<void> {
    // 1. 60Hz Viseme lip-sync ticks
    const unlistenViseme = await TauriBridge.listen('viseme_tick', (payload: any) => {
      this.vrmViewer.onVisemeTick(payload);
    });
    this.unlisteners.push(unlistenViseme);

    // 2. Subtitle update & conversational state
    const unlistenSubtitle = await TauriBridge.listen('subtitle_update', (payload: any) => {
      if (!payload) return;
      const stateStr = String(payload.state || 'idle').toLowerCase();
      this.updateSubtitle(stateStr, payload.text);

      // Parse implicit kinematic cues and gestures from text
      if (payload.text) {
        const cue = parseKinematicCues(payload.text);
        if (cue.gesture) {
          this.vrmViewer.playGesture(cue.gesture);
        }
        if (cue.emotion) {
          this.vrmViewer.setEmotion(cue.emotion, 1.0);
        }
      }
    });
    this.unlisteners.push(unlistenSubtitle);

    // 2b. Thought scratchpad event broadcast
    const unlistenThought = await TauriBridge.listen('thought_event', (payload: any) => {
      if (!payload) return;
      if (this.el.thoughtCard) {
        this.el.thoughtCard.hidden = false;
        if (!payload.is_done) {
          this.el.thoughtCard.open = true;
          if (this.el.thoughtBadge) {
            this.el.thoughtBadge.textContent = 'Thinking...';
          }
        } else {
          if (this.el.thoughtBadge) {
            this.el.thoughtBadge.textContent = 'Thought complete';
          }
        }
      }
      if (this.el.thoughtText && payload.text) {
        this.el.thoughtText.textContent = (this.el.thoughtText.textContent || '') + payload.text;
      }
    });
    this.unlisteners.push(unlistenThought);

    // 3. Audible Speech Synthesis event (Native Soundcard First + Automatic Fallback)
    const unlistenSpeak = await TauriBridge.listen('speak_sentence', (payload: any) => {
      // If native CPAL playback is handling audio, suppress browser speech synthesis to eliminate double speech
      if (payload?.native_audio) {
        return;
      }
      if (payload?.text && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        try {
          const utter = new SpeechSynthesisUtterance(payload.text);
          utter.lang = payload.language || 'vi-VN';
          utter.rate = Number(payload.rate || 1.0);
          this.enqueueSpeech(utter);
        } catch (e) {
          console.warn('Speech synthesis error:', e);
        }
      }
    });
    this.unlisteners.push(unlistenSpeak);

    // 4. Gesture event broadcast
    const unlistenGesture = await TauriBridge.listen('gesture_event', (payload: any) => {
      if (payload?.name) {
        this.vrmViewer.playGesture(payload.name, payload.duration);
      }
    });
    this.unlisteners.push(unlistenGesture);

    // 5. Emotion event broadcast
    const unlistenEmotion = await TauriBridge.listen('emotion_event', (payload: any) => {
      if (payload?.name) {
        this.vrmViewer.setEmotion(payload.name, payload.weight ?? 1.0);
      }
    });
    this.unlisteners.push(unlistenEmotion);

    // 6. Audio RMS state event
    const unlistenAudio = await TauriBridge.listen('audio_state', (payload: any) => {
      if (payload) {
        const rms = Number(payload.rms_level || 0);
        if (this.el.btnMic) {
          const scale = 1 + Math.min(0.4, rms * 4);
          this.el.btnMic.style.transform = this.isMicActive ? `scale(${scale})` : '';
        }
        if (payload.barge_in_triggered) {
          this.cancelSpeech();
        }
      }
    });
    this.unlisteners.push(unlistenAudio);
  }

  private initSpeechRecognition(): void {
    if (typeof window === 'undefined') return;
    const SpeechRec = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRec) {
      try {
        this.speechRec = new SpeechRec();
        this.speechRec.continuous = false;
        this.speechRec.interimResults = true;
        this.speechRec.onresult = (ev: any) => {
          let transcript = '';
          for (let i = ev.resultIndex; i < ev.results.length; i++) {
            transcript += ev.results[i][0].transcript;
          }
          if (transcript.trim()) {
            this.el.chatInput.value = transcript;
            this.updateSubtitle('listening', transcript);
          }
        };
        this.speechRec.onend = () => {
          if (this.isMicActive) {
            const text = this.el.chatInput.value.trim();
            if (text) {
              this.handleSendMessage();
            }
          }
        };
      } catch {
        this.speechRec = null;
      }
    }
  }

  private bindDomEvents(): void {
    const e = this.el;

    // Send Message on click or Enter (safeguarding Vietnamese Telex/VNI IME composition)
    let isComposing = false;
    e.chatInput.addEventListener('compositionstart', () => {
      isComposing = true;
    });
    e.chatInput.addEventListener('compositionend', () => {
      isComposing = false;
    });
    e.btnSend.addEventListener('click', () => this.handleSendMessage());
    e.chatInput.addEventListener('keydown', (ev: KeyboardEvent) => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing && !isComposing && ev.keyCode !== 229) {
        ev.preventDefault();
        this.handleSendMessage();
      }
    });

    // Tutor Mode dropdown
    e.tutorSelect.addEventListener('change', (ev: Event) => {
      const mode = (ev.target as HTMLSelectElement).value;
      this.handleTutorModeChange(mode);
    });

    // Top action buttons
    e.btnGestures.addEventListener('click', () => this.toggleGesturePanel());
    e.btnKnowledge.addEventListener('click', () => this.toggleDrawer('knowledge'));
    e.btnSettings.addEventListener('click', () => this.toggleDrawer('settings'));
    if (e.btnFlashcards) {
      e.btnFlashcards.addEventListener('click', () => this.toggleDrawer('flashcards'));
    }
    if (e.btnPronounce) {
      e.btnPronounce.addEventListener('click', () => this.openPronounceModal());
    }

    // Close buttons
    e.btnCloseKnowledge.addEventListener('click', () => this.closeDrawer());
    e.btnCloseSettings.addEventListener('click', () => this.closeDrawer());
    if (e.btnCloseFlashcards) {
      e.btnCloseFlashcards.addEventListener('click', () => this.closeDrawer());
    }
    if (e.btnClosePronounce) {
      e.btnClosePronounce.addEventListener('click', () => this.closePronounceModal());
    }
    e.scrim.addEventListener('click', () => this.closeDrawer());

    e.btnResetCamera.addEventListener('click', () => {
      this.vrmViewer.resetCamera();
      this.showToast('Góc nhìn đã được căn chỉnh lại');
    });

    // Gesture grid buttons
    e.gesturePanel.addEventListener('click', (ev: MouseEvent) => {
      const target = (ev.target as HTMLElement).closest('[data-gesture]') as HTMLElement | null;
      if (target?.dataset.gesture) {
        this.handleTriggerGesture(target.dataset.gesture);
      }
    });

    // Settings slider outputs
    e.temperature.addEventListener('input', () => {
      e.outTemperature.textContent = Number(e.temperature.value).toFixed(2);
    });
    e.rate.addEventListener('input', () => {
      e.outRate.textContent = Number(e.rate.value).toFixed(2);
    });

    // Save settings
    e.btnSave.addEventListener('click', () => this.handleSaveSettings());

    // Knowledge base document actions
    e.btnIngest.addEventListener('click', () => this.handleIngestDoc());
    this.setupDropZone(e.docDrop, e.docFile, (file) => this.readDocumentFile(file));
    this.setupDropZone(e.vrmDrop, e.vrmFile, (file) => this.loadVrmFile(file));

    // Flashcards tabs & actions
    this.bindFlashcardEvents();

    // Pronunciation assessment actions
    if (e.btnPronounceAssess) {
      e.btnPronounceAssess.addEventListener('click', () => this.handleAssessPronunciation());
    }

    // Mic interaction (push-to-talk + click toggle)
    this.setupMic();

    // Mouse tracking for avatar LookAt
    window.addEventListener(
      'pointermove',
      (ev) => {
        const nx = (ev.clientX / window.innerWidth) * 2 - 1;
        const ny = (ev.clientY / window.innerHeight) * 2 - 1;
        this.vrmViewer.updateCursor(nx, ny);
      },
      { passive: true }
    );
    document.documentElement.addEventListener('pointerleave', () => {
      this.vrmViewer.releaseCursor();
    });

    // Push-to-talk with Space bar
    window.addEventListener('keydown', (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        this.closeDrawer();
        this.closePronounceModal();
      }
      if (ev.code === 'Space' && !ev.repeat && this.canPushToTalkKey(ev)) {
        ev.preventDefault();
        this.spaceHeld = true;
        this.setMicActive(true);
      }
    });
    window.addEventListener('keyup', (ev: KeyboardEvent) => {
      if (ev.code === 'Space' && this.spaceHeld) {
        this.spaceHeld = false;
        this.setMicActive(false);
      }
    });

    // File drop over window for VRM loading
    window.addEventListener('dragover', (ev) => ev.preventDefault());
    window.addEventListener('drop', (ev) => {
      ev.preventDefault();
      const file = ev.dataTransfer?.files?.[0];
      if (file && /\.vrm$/i.test(file.name) && !(ev.target as HTMLElement).closest?.('.drop-zone')) {
        this.loadVrmFile(file);
      }
    });
  }

  private bindFlashcardEvents(): void {
    const e = this.el;
    if (!e.flashcardsDrawer) return;

    // Tabs
    const tabs = e.flashcardsDrawer.querySelectorAll('.drawer-tab');
    tabs.forEach((tab: Element) => {
      tab.addEventListener('click', () => {
        const tabEl = tab as HTMLElement;
        const tabKey = tabEl.dataset.tab;
        tabs.forEach((t: Element) => t.classList.remove('active'));
        tabEl.classList.add('active');
        e.flashcardsDrawer.querySelectorAll('.tab-content').forEach((tc: Element) => {
          const tcEl = tc as HTMLElement;
          tcEl.classList.toggle('active', tcEl.id === `tab-fc-${tabKey}`);
        });
      });
    });

    // Reveal answer
    if (e.btnFcReveal) {
      e.btnFcReveal.addEventListener('click', () => this.handleRevealAnswer());
    }

    // Rating buttons
    if (e.fcRatingActions) {
      e.fcRatingActions.addEventListener('click', (ev: MouseEvent) => {
        const btn = (ev.target as HTMLElement).closest('[data-rating]') as HTMLElement | null;
        if (btn?.dataset.rating) {
          this.handleReviewCard(btn.dataset.rating);
        }
      });
    }

    // Create card
    if (e.btnFcSave) {
      e.btnFcSave.addEventListener('click', () => this.handleCreateCard());
    }
  }

  private canPushToTalkKey(ev: KeyboardEvent): boolean {
    const t = ev.target;
    const isTyping =
      t instanceof HTMLElement &&
      (t.matches('input, textarea, select, button, [contenteditable]') ||
        Boolean(t.closest('.drawer')) ||
        Boolean(t.closest('dialog')));
    return !isTyping && !this.openDrawerName;
  }

  private setupMic(): void {
    const btn = this.el.btnMic;

    btn.addEventListener('pointerdown', (ev: PointerEvent) => {
      ev.preventDefault();
      btn.setPointerCapture?.(ev.pointerId);
      this.micDownAt = performance.now();
      if (this.isMicActive) {
        this.setMicActive(false);
      } else {
        this.setMicActive(true);
      }
    });

    const stopHold = () => {
      if (performance.now() - this.micDownAt >= HOLD_MS && this.isMicActive) {
        this.setMicActive(false);
      }
    };
    btn.addEventListener('pointerup', stopHold);
    btn.addEventListener('pointercancel', stopHold);
  }

  private async setMicActive(active: boolean): Promise<void> {
    this.isMicActive = active;
    this.el.btnMic.classList.toggle('listening', active);
    this.el.btnMic.setAttribute('aria-pressed', String(active));

    // Handle Web Speech API
    if (this.speechRec) {
      try {
        if (active) {
          const preset = TUTOR_PRESETS[this.currentModeKey];
          this.speechRec.lang = preset?.language || 'vi-VN';
          this.speechRec.start();
        } else {
          this.speechRec.stop();
        }
      } catch {
        // ignore
      }
    }

    // Handle native audio capture in Tauri
    try {
      await TauriBridge.invoke('toggle_mic', { active });
      if (active) {
        this.setStatus('listening');
        this.vrmViewer.setState('listening');
      } else if (this.vrmViewer.state === 'listening') {
        this.setStatus('idle');
        this.vrmViewer.setState('idle');
      }
    } catch (err: any) {
      console.error('Error toggling mic:', err);
      this.showToast(`Lỗi micro: ${err.message || err}`);
    }
  }

  public setStatus(state: string): void {
    const norm = state.toLowerCase();
    this.el.subtitleCard.dataset.state = norm;
    this.el.subtitleBadge.textContent = `Vira · ${STATUS_LABELS[norm] || norm}`;
    this.vrmViewer.setState(norm);
  }

  public updateSubtitle(state: string, text?: string): void {
    this.el.subtitleCard.classList.add('active');
    this.setStatus(state);
    if (text !== undefined && text !== '') {
      this.el.subtitleText.textContent = text;
      // Pre-fill pronunciation input with last spoken tutor phrase
      if (this.el.pronounceTarget && state === 'speaking') {
        this.el.pronounceTarget.value = text;
      }
    }
  }

  private enqueueSpeech(utter: SpeechSynthesisUtterance): void {
    const handleNext = () => {
      this.isSpeakingSpeech = false;
      this.playNextSpeech();
    };
    utter.onend = handleNext;
    utter.onerror = handleNext;
    this.speechQueue.push(utter);
    if (!this.isSpeakingSpeech) {
      this.playNextSpeech();
    }
  }

  private playNextSpeech(): void {
    if (this.speechQueue.length === 0) {
      this.isSpeakingSpeech = false;
      return;
    }
    const next = this.speechQueue.shift();
    if (next && typeof window !== 'undefined' && 'speechSynthesis' in window) {
      this.isSpeakingSpeech = true;
      try {
        window.speechSynthesis.speak(next);
      } catch (e) {
        console.warn('Speech synthesis speak error:', e);
        this.isSpeakingSpeech = false;
        this.playNextSpeech();
      }
    }
  }

  public cancelSpeech(): void {
    this.speechQueue = [];
    this.isSpeakingSpeech = false;
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (e) {
        console.warn('Speech synthesis cancel error:', e);
      }
    }
  }

  public async handleSendMessage(): Promise<void> {
    const text = this.el.chatInput.value.trim();
    if (!text) return;

    this.el.chatInput.value = '';
    this.setMicActive(false);
    this.updateSubtitle('thinking', text);

    if (this.el.thoughtText) {
      this.el.thoughtText.textContent = '';
    }
    if (this.el.thoughtBadge) {
      this.el.thoughtBadge.textContent = 'Thinking...';
    }
    if (this.el.thoughtCard) {
      this.el.thoughtCard.open = false;
      this.el.thoughtCard.hidden = true;
    }

    this.cancelSpeech();

    try {
      await TauriBridge.invoke('send_message', { payload: { text } });
    } catch (err: any) {
      this.updateSubtitle('error', `Lỗi xử lý: ${err.message || err}`);
    }
  }

  public async handleTriggerGesture(name: string): Promise<void> {
    this.vrmViewer.playGesture(name);
    try {
      await TauriBridge.invoke('trigger_gesture', { name, duration: 2.5 });
    } catch {
      // ignore
    }
  }

  public async handleTutorModeChange(modeKey: string): Promise<void> {
    const norm = normalizeTutorMode(modeKey);
    this.setTutorModeUI(norm);
    try {
      await TauriBridge.invoke('set_tutor_mode', { mode: norm });
      this.showToast(`Đã chuyển sang chế độ ${TUTOR_PRESETS[norm]?.name || norm}`);
    } catch (err: any) {
      this.showToast(`Không thể đổi chế độ: ${err.message || err}`);
    }
  }

  private setTutorModeUI(modeKey: string): void {
    const norm = normalizeTutorMode(modeKey);
    this.currentModeKey = norm;
    const preset = TUTOR_PRESETS[norm] || TUTOR_PRESETS.general_assistant;
    this.el.tutorSelect.value = preset.id;
    this.el.brandModeLabel.textContent = preset.name;
    this.el.systemPrompt.value = preset.systemPrompt;
  }

  private toggleGesturePanel(): void {
    const isHidden = this.el.gesturePanel.hidden;
    this.el.gesturePanel.hidden = !isHidden;
    this.el.btnGestures.setAttribute('aria-expanded', String(isHidden));
  }

  private toggleDrawer(name: 'knowledge' | 'settings' | 'flashcards'): void {
    if (this.openDrawerName === name) {
      this.closeDrawer();
      return;
    }
    this.closeDrawer();
    this.openDrawerName = name;
    let drawer: HTMLElement | null = null;
    if (name === 'knowledge') drawer = this.el.knowledgeDrawer;
    else if (name === 'settings') drawer = this.el.settingsDrawer;
    else if (name === 'flashcards') drawer = this.el.flashcardsDrawer;

    if (drawer) {
      drawer.classList.add('open');
      drawer.setAttribute('aria-hidden', 'false');
      drawer.removeAttribute('inert');
      this.el.scrim.hidden = false;
    }
  }

  private closeDrawer(): void {
    if (!this.openDrawerName) return;
    let drawer: HTMLElement | null = null;
    if (this.openDrawerName === 'knowledge') drawer = this.el.knowledgeDrawer;
    else if (this.openDrawerName === 'settings') drawer = this.el.settingsDrawer;
    else if (this.openDrawerName === 'flashcards') drawer = this.el.flashcardsDrawer;

    if (drawer) {
      drawer.classList.remove('open');
      drawer.setAttribute('aria-hidden', 'true');
      drawer.setAttribute('inert', '');
    }
    this.el.scrim.hidden = true;
    this.openDrawerName = null;
  }

  private openPronounceModal(): void {
    if (this.el.pronounceModal) {
      if (typeof this.el.pronounceModal.showModal === 'function') {
        this.el.pronounceModal.showModal();
      } else {
        this.el.pronounceModal.setAttribute('open', '');
      }
    }
  }

  private closePronounceModal(): void {
    if (this.el.pronounceModal) {
      if (typeof this.el.pronounceModal.close === 'function') {
        this.el.pronounceModal.close();
      } else {
        this.el.pronounceModal.removeAttribute('open');
      }
    }
  }

  private async handleSaveSettings(): Promise<void> {
    const e = this.el;
    const payload = {
      llm_endpoint: e.baseUrl.value.trim() || undefined,
      llm_model: e.model.value.trim() || undefined,
      llm_api_key: e.apiKey.value.trim() || undefined,
      temperature: Number(e.temperature.value),
      tutor_mode: this.currentModeKey,
      system_prompt: e.systemPrompt.value.trim() || undefined,
      tts_voice: e.voiceSelect.value || undefined,
      tts_speed: Number(e.rate.value),
      rag_enabled: e.rag.checked,
    };

    try {
      await TauriBridge.invoke('save_settings', { settings: payload });
      this.showToast('Cài đặt đã được lưu thành công');
      this.closeDrawer();
    } catch (err: any) {
      this.showToast(`Lỗi lưu cài đặt: ${err.message || err}`);
    }
  }

  private setupDropZone(zone: HTMLElement, input: HTMLInputElement, onFile: (file: File) => void): void {
    if (!zone || !input) return;
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', (ev: KeyboardEvent) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        input.click();
      }
    });
    zone.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (ev) => {
      ev.preventDefault();
      zone.classList.remove('dragover');
      const file = ev.dataTransfer?.files?.[0];
      if (file) onFile(file);
    });
    input.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) onFile(file);
      input.value = '';
    });
  }

  private readDocumentFile(file: File): void {
    if (!/\.(txt|md)$/i.test(file.name)) {
      this.showToast('Vui lòng chọn tệp định dạng .txt hoặc .md');
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      this.el.docTitle.value = file.name.replace(/\.[^/.]+$/, '');
      this.el.docContent.value = String(reader.result || '');
      this.showToast(`Đã nạp nội dung: ${file.name}`);
    };
    reader.onerror = () => this.showToast('Không thể đọc tệp');
    reader.readAsText(file);
  }

  // ============================================================================
  // Knowledge Base (RAG) Actions
  // ============================================================================

  private async handleIngestDoc(): Promise<void> {
    const title = this.el.docTitle.value.trim();
    const content = this.el.docContent.value.trim();
    if (!title || !content) {
      this.showToast('Vui lòng nhập cả tiêu đề và nội dung');
      return;
    }

    try {
      await TauriBridge.invoke('ingest_document', {
        payload: { title, content },
      });
      this.el.docTitle.value = '';
      this.el.docContent.value = '';
      await this.refreshDocumentList();
      this.showToast(`Đã lưu "${title}" vào bộ nhớ kiến thức (RAG)`);
    } catch (err: any) {
      this.showToast(`Lỗi thêm tài liệu: ${err.message || err}`);
    }
  }

  private async refreshDocumentList(): Promise<void> {
    const box = this.el.docList;
    if (!box) return;
    box.replaceChildren();

    let docs: Array<any> = [];
    try {
      docs = await TauriBridge.invoke('list_documents');
    } catch {
      docs = [];
    }

    if (!docs || !docs.length) {
      const empty = document.createElement('div');
      empty.className = 'doc-empty';
      empty.textContent = 'Chưa có tài liệu nào.';
      box.appendChild(empty);
      return;
    }

    for (const doc of docs) {
      const card = document.createElement('div');
      card.className = 'doc-card';

      const info = document.createElement('div');
      info.className = 'doc-info';

      const title = document.createElement('div');
      title.className = 'doc-title';
      title.textContent = doc.title;

      const meta = document.createElement('div');
      meta.className = 'doc-meta';
      const count = doc.chunk_count ? `${doc.chunk_count} phân đoạn · ` : '';
      meta.textContent = `${count}${doc.content?.length || 0} ký tự`;
      info.append(title, meta);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn-delete';
      del.title = 'Xóa tài liệu';
      del.innerHTML =
        '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
      del.addEventListener('click', async () => {
        try {
          await TauriBridge.invoke('delete_document', { doc_id: doc.id });
          await this.refreshDocumentList();
          this.showToast(`Đã xóa "${doc.title}"`);
        } catch (err: any) {
          this.showToast(`Lỗi xóa: ${err.message || err}`);
        }
      });

      card.append(info, del);
      box.appendChild(card);
    }
  }

  // ============================================================================
  // Flashcards (FSRS) Actions
  // ============================================================================

  public async refreshFlashcards(): Promise<void> {
    const e = this.el;
    if (!e.flashcardsDrawer) return;

    try {
      const allCards = (await TauriBridge.invoke('list_flashcards')) || [];
      const due = (await TauriBridge.invoke('get_due_flashcards')) || [];
      this.dueCards = due;

      if (e.fcTotalCount) e.fcTotalCount.textContent = String(allCards.length);
      if (e.fcDuePill) e.fcDuePill.textContent = `${due.length} thẻ cần ôn`;
      if (e.dueBadge) {
        e.dueBadge.textContent = String(due.length);
        e.dueBadge.hidden = due.length === 0;
      }

      this.renderFlashcardsList(allCards);
      this.presentNextDueCard();
    } catch {
      // ignore
    }
  }

  private presentNextDueCard(): void {
    const e = this.el;
    if (!this.dueCards.length) {
      this.currentStudyCard = null;
      if (e.fcCardFront) e.fcCardFront.textContent = '🎉 Tuyệt vời! Bạn đã hoàn thành tất cả thẻ ôn tập hôm nay!';
      if (e.fcCardBack) e.fcCardBack.hidden = true;
      if (e.btnFcReveal) e.btnFcReveal.hidden = true;
      if (e.fcRatingActions) e.fcRatingActions.hidden = true;
      return;
    }

    this.currentStudyCard = this.dueCards[0];
    if (e.fcCardLevel) e.fcCardLevel.textContent = this.currentStudyCard.cefr_level || 'A1';
    if (e.fcCardFront) e.fcCardFront.textContent = this.currentStudyCard.front;
    if (e.fcCardBack) {
      e.fcCardBack.textContent = this.currentStudyCard.back;
      e.fcCardBack.hidden = true;
    }
    if (e.btnFcReveal) e.btnFcReveal.hidden = false;
    if (e.fcRatingActions) e.fcRatingActions.hidden = true;
  }

  private handleRevealAnswer(): void {
    if (!this.currentStudyCard) return;
    this.el.fcCardBack.hidden = false;
    this.el.btnFcReveal.hidden = true;
    this.el.fcRatingActions.hidden = false;
  }

  private async handleReviewCard(rating: string): Promise<void> {
    if (!this.currentStudyCard) return;
    try {
      await TauriBridge.invoke('review_flashcard', {
        payload: {
          card_id: this.currentStudyCard.id,
          rating,
        },
      });
      this.dueCards.shift();
      this.presentNextDueCard();
      await this.refreshFlashcards();
    } catch (err: any) {
      this.showToast(`Lỗi lưu đánh giá: ${err.message || err}`);
    }
  }

  private async handleCreateCard(): Promise<void> {
    const e = this.el;
    const front = e.fcInputFront.value.trim();
    const back = e.fcInputBack.value.trim();
    const cefr_level = e.fcSelectLevel.value;

    if (!front || !back) {
      this.showToast('Vui lòng nhập cả mặt trước và mặt sau thẻ');
      return;
    }

    try {
      await TauriBridge.invoke('create_flashcard', {
        payload: { front, back, cefr_level },
      });
      e.fcInputFront.value = '';
      e.fcInputBack.value = '';
      await this.refreshFlashcards();
      this.showToast(`Đã tạo flashcard "${front}"`);
    } catch (err: any) {
      this.showToast(`Lỗi tạo thẻ: ${err.message || err}`);
    }
  }

  private renderFlashcardsList(cards: Array<any>): void {
    const box = this.el.fcListContainer;
    if (!box) return;
    box.replaceChildren();

    if (!cards.length) {
      const empty = document.createElement('div');
      empty.className = 'doc-empty';
      empty.textContent = 'Chưa có thẻ từ vựng nào.';
      box.appendChild(empty);
      return;
    }

    for (const card of cards) {
      const row = document.createElement('div');
      row.className = 'doc-card';

      const info = document.createElement('div');
      info.className = 'doc-info';

      const front = document.createElement('div');
      front.className = 'doc-title';
      front.textContent = card.front;

      const meta = document.createElement('div');
      meta.className = 'doc-meta';
      meta.textContent = `${card.back} · [${card.cefr_level || 'A1'}] · Đã ôn ${card.reps || 0} lần`;

      info.append(front, meta);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn-delete';
      del.title = 'Xóa thẻ';
      del.innerHTML =
        '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
      del.addEventListener('click', async () => {
        try {
          await TauriBridge.invoke('delete_flashcard', { card_id: card.id });
          await this.refreshFlashcards();
          this.showToast(`Đã xóa thẻ "${card.front}"`);
        } catch (err: any) {
          this.showToast(`Lỗi xóa: ${err.message || err}`);
        }
      });

      row.append(info, del);
      box.appendChild(row);
    }
  }

  // ============================================================================
  // Pronunciation Assessment (GOP) Actions
  // ============================================================================

  private async handleAssessPronunciation(): Promise<void> {
    const target = this.el.pronounceTarget.value.trim();
    if (!target) {
      this.showToast('Vui lòng nhập câu cần luyện phát âm');
      return;
    }

    this.el.btnPronounceAssess.disabled = true;
    this.el.btnPronounceAssess.textContent = 'Đang phân tích âm học GOP…';

    try {
      const result = await TauriBridge.invoke('assess_pronunciation', {
        payload: { target_text: target },
      });

      this.el.pronounceResult.hidden = false;
      const score = Math.round(result.overall_score || 0);
      this.el.pronounceOverallScore.textContent = String(score);

      const status = this.el.pronounceAccuracyLabel;
      if (score >= 80) {
        status.textContent = 'Xuất sắc · Chuẩn giọng bản ngữ';
        status.className = 'score-status status-correct';
      } else if (score >= 55) {
        status.textContent = 'Khá tốt · Cần chú ý âm sắc';
        status.className = 'score-status status-flaw';
      } else {
        status.textContent = 'Cần luyện tập thêm';
        status.className = 'score-status status-mispronounced';
      }

      // Render phoneme chips
      const container = this.el.pronouncePhonemes;
      container.replaceChildren();

      const scores = result.phoneme_scores || [];
      for (const p of scores) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `phoneme-chip accuracy-${String(p.accuracy).toLowerCase()}`;
        chip.innerHTML = `<span class="ph-symbol">${p.phoneme}</span><span class="ph-score">${Math.round(p.score)}%</span>`;
        chip.title = `Âm [${p.phoneme}] - Điểm: ${Math.round(p.score)}%. Nhấn để xem khẩu hình trên 3D avatar`;

        chip.addEventListener('click', () => {
          if (p.viseme_hint) {
            this.vrmViewer.onVisemeTick({
              weights: p.viseme_hint,
              is_speaking: true,
            });
            this.showToast(`Khẩu hình cho âm [${p.phoneme}] (${p.recommended_viseme || 'aa'})`);
          }
        });

        container.appendChild(chip);
      }
    } catch (err: any) {
      this.showToast(`Lỗi đánh giá phát âm: ${err.message || err}`);
    } finally {
      this.el.btnPronounceAssess.disabled = false;
      this.el.btnPronounceAssess.textContent = 'Ghi âm & Đánh giá phát âm';
    }
  }

  public async loadVrmFile(file: File): Promise<void> {
    if (!/\.vrm$/i.test(file.name)) {
      this.showToast('Vui lòng chọn tệp .vrm');
      return;
    }
    const label = this.el.vrmDrop.querySelector('.drop-zone-text');
    if (label) label.textContent = `Đang tải ${file.name}…`;

    try {
      await this.vrmViewer.loadModel(await file.arrayBuffer());
      if (label) label.textContent = `Đang dùng: ${file.name}`;
      this.showToast(`Avatar 3D "${file.name}" đã được tải thành công`);
    } catch (err: any) {
      if (label) label.textContent = 'Thả tệp .vrm hoặc nhấn để tải avatar mới';
      this.showToast(`Không thể tải VRM: ${err.message || err}`);
    }
  }

  public showToast(message: string, ms = 3400): void {
    clearTimeout(this.toastTimer);
    const toast = this.el.toast;
    toast.textContent = message;
    toast.classList.add('show');
    this.toastTimer = setTimeout(() => toast.classList.remove('show'), ms);
  }

  public dispose(): void {
    this.cancelSpeech();
    for (const unlisten of this.unlisteners) {
      unlisten();
    }
    this.unlisteners = [];
  }
}
