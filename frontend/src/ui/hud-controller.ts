import { VRMViewer } from '../stage/vrm-viewer.js';
import { TauriBridge } from '../ipc/tauri-bridge.js';

export interface TutorPreset {
  id: string;
  name: string;
  language: string;
  systemPrompt: string;
}

export const TUTOR_PRESETS: Record<string, TutorPreset> = {
  vietnameseTutor: {
    id: 'vietnameseTutor',
    name: 'Vietnamese Tutor',
    language: 'vi-VN',
    systemPrompt:
      'Bạn là Vira, gia sư tiếng Việt thông thái, dịu dàng và nhiệt huyết. Bạn hướng dẫn người học từ phát âm, dấu thanh, từ vựng đến ngữ pháp phong phú của tiếng Việt.',
  },
  englishTutor: {
    id: 'englishTutor',
    name: 'English Tutor',
    language: 'en-US',
    systemPrompt:
      'You are Vira, an encouraging, patient English tutor. Help the user master natural conversation, vocabulary, and grammar with gentle corrections.',
  },
  japaneseTutor: {
    id: 'japaneseTutor',
    name: 'Japanese Tutor',
    language: 'ja-JP',
    systemPrompt:
      'あなたはViraです。親切で丁寧な日本語の先生です。学習者の日本語の会話や文法を優しくサポートしてください。',
  },
  general: {
    id: 'general',
    name: 'General Assistant',
    language: 'vi-VN',
    systemPrompt:
      'Bạn là Vira, trợ lý ảo 3D thông minh, chu đáo và thân thiện. Bạn luôn sẵn sàng giải đáp mọi thắc mắc của người dùng.',
  },
};

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
  private currentModeKey = 'vietnameseTutor';
  private toastTimer: any = null;
  private openDrawerName: string | null = null;
  private micDownAt = 0;
  private isMicActive = false;
  private spaceHeld = false;
  private unlisteners: Array<() => void> = [];

  constructor(vrmViewer: VRMViewer) {
    this.vrmViewer = vrmViewer;
  }

  public async init(): Promise<void> {
    this.cacheElements();
    this.bindDomEvents();
    await this.setupIpcListeners();
    this.refreshDocumentList();
    this.setStatus('idle');

    // Sync initial state if available
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
      tutorSelect: $('tutor-select') as HTMLSelectElement,
      brandModeLabel: $('brand-mode-label') as HTMLElement,
      btnGestures: $('btn-gestures') as HTMLButtonElement,
      gesturePanel: $('gesture-panel') as HTMLElement,
      btnKnowledge: $('btn-knowledge-toggle') as HTMLButtonElement,
      btnCloseKnowledge: $('btn-close-knowledge') as HTMLButtonElement,
      knowledgeDrawer: $('knowledge-drawer') as HTMLElement,
      docTitle: $('doc-title-input') as HTMLInputElement,
      docContent: $('doc-content-input') as HTMLTextAreaElement,
      docDrop: $('doc-drop-zone') as HTMLElement,
      docFile: $('doc-file-input') as HTMLInputElement,
      btnIngest: $('btn-ingest-doc') as HTMLButtonElement,
      docList: $('doc-list-container') as HTMLElement,
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
      btnResetCamera: $('btn-reset-camera') as HTMLButtonElement,
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
    });
    this.unlisteners.push(unlistenSubtitle);

    // 3. Gesture event broadcast
    const unlistenGesture = await TauriBridge.listen('gesture_event', (payload: any) => {
      if (payload?.name) {
        this.vrmViewer.playGesture(payload.name, payload.duration);
      }
    });
    this.unlisteners.push(unlistenGesture);

    // 4. Emotion event broadcast
    const unlistenEmotion = await TauriBridge.listen('emotion_event', (payload: any) => {
      if (payload?.name) {
        this.vrmViewer.setEmotion(payload.name, payload.weight ?? 1.0);
      }
    });
    this.unlisteners.push(unlistenEmotion);
  }

  private bindDomEvents(): void {
    const e = this.el;

    // Send Message on click or Enter (safeguarding Vietnamese Telex/VNI IME composition)
    e.btnSend.addEventListener('click', () => this.handleSendMessage());
    e.chatInput.addEventListener('keydown', (ev: KeyboardEvent) => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
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
    e.btnCloseKnowledge.addEventListener('click', () => this.closeDrawer());
    e.btnCloseSettings.addEventListener('click', () => this.closeDrawer());
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
      if (ev.key === 'Escape') this.closeDrawer();
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

  private canPushToTalkKey(ev: KeyboardEvent): boolean {
    const t = ev.target;
    const isTyping =
      t instanceof HTMLElement &&
      (t.matches('input, textarea, select, button, [contenteditable]') || Boolean(t.closest('.drawer')));
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

    btn.addEventListener('pointerup', () => {
      if (performance.now() - this.micDownAt >= HOLD_MS && this.isMicActive) {
        this.setMicActive(false);
      }
    });
  }

  private async setMicActive(active: boolean): Promise<void> {
    this.isMicActive = active;
    this.el.btnMic.classList.toggle('listening', active);
    this.el.btnMic.setAttribute('aria-pressed', String(active));

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
    }
  }

  public async handleSendMessage(): Promise<void> {
    const text = this.el.chatInput.value.trim();
    if (!text) return;

    this.el.chatInput.value = '';
    this.setMicActive(false);
    this.updateSubtitle('thinking', text);

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
    this.setTutorModeUI(modeKey);
    try {
      await TauriBridge.invoke('set_tutor_mode', { mode: modeKey });
      this.showToast(`Đã chuyển sang chế độ ${TUTOR_PRESETS[modeKey]?.name || modeKey}`);
    } catch (err: any) {
      this.showToast(`Không thể đổi chế độ: ${err.message || err}`);
    }
  }

  private setTutorModeUI(modeKey: string): void {
    this.currentModeKey = modeKey;
    const preset = TUTOR_PRESETS[modeKey] || TUTOR_PRESETS.general;
    this.el.tutorSelect.value = modeKey;
    this.el.brandModeLabel.textContent = preset.name;
    this.el.systemPrompt.value = preset.systemPrompt;
  }

  private toggleGesturePanel(): void {
    const isHidden = this.el.gesturePanel.hidden;
    this.el.gesturePanel.hidden = !isHidden;
    this.el.btnGestures.setAttribute('aria-expanded', String(isHidden));
  }

  private toggleDrawer(name: 'knowledge' | 'settings'): void {
    if (this.openDrawerName === name) {
      this.closeDrawer();
      return;
    }
    this.closeDrawer();
    this.openDrawerName = name;
    const drawer = name === 'knowledge' ? this.el.knowledgeDrawer : this.el.settingsDrawer;
    drawer.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    drawer.removeAttribute('inert');
    this.el.scrim.hidden = false;
    (name === 'knowledge' ? this.el.btnCloseKnowledge : this.el.btnCloseSettings).focus();
  }

  private closeDrawer(): void {
    if (!this.openDrawerName) return;
    const drawer = this.openDrawerName === 'knowledge' ? this.el.knowledgeDrawer : this.el.settingsDrawer;
    drawer.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
    drawer.setAttribute('inert', '');
    this.el.scrim.hidden = true;
    this.openDrawerName = null;
  }

  private async handleSaveSettings(): Promise<void> {
    const e = this.el;
    const payload = {
      llm_base_url: e.baseUrl.value.trim() || undefined,
      llm_model: e.model.value.trim() || undefined,
      llm_api_key: e.apiKey.value.trim() || undefined,
      temperature: Number(e.temperature.value),
      tutor_mode: this.currentModeKey,
      system_prompt: e.systemPrompt.value.trim() || undefined,
      tts_voice: e.voiceSelect.value || undefined,
      speech_rate: Number(e.rate.value),
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

  private handleIngestDoc(): void {
    const title = this.el.docTitle.value.trim();
    const content = this.el.docContent.value.trim();
    if (!title || !content) {
      this.showToast('Vui lòng nhập cả tiêu đề và nội dung');
      return;
    }

    const docs = this.getStoredDocs();
    docs.push({
      id: 'doc_' + Date.now(),
      title,
      content,
      createdAt: new Date().toISOString(),
    });
    localStorage.setItem('vira_docs', JSON.stringify(docs));

    this.el.docTitle.value = '';
    this.el.docContent.value = '';
    this.refreshDocumentList();
    this.showToast(`Đã lưu "${title}" vào bộ nhớ`);
  }

  private getStoredDocs(): Array<{ id: string; title: string; content: string; createdAt: string }> {
    try {
      const raw = localStorage.getItem('vira_docs');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  private refreshDocumentList(): void {
    const box = this.el.docList;
    box.replaceChildren();
    const docs = this.getStoredDocs();

    if (!docs.length) {
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
      meta.textContent = `${new Date(doc.createdAt).toLocaleDateString()} · ${doc.content.length} ký tự`;
      info.append(title, meta);

      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn-delete';
      del.title = 'Xóa tài liệu';
      del.innerHTML =
        '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
      del.addEventListener('click', () => {
        const remaining = this.getStoredDocs().filter((d) => d.id !== doc.id);
        localStorage.setItem('vira_docs', JSON.stringify(remaining));
        this.refreshDocumentList();
        this.showToast(`Đã xóa "${doc.title}"`);
      });

      card.append(info, del);
      box.appendChild(card);
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
    for (const unlisten of this.unlisteners) {
      unlisten();
    }
    this.unlisteners = [];
  }
}
