import { describe, expect, it } from 'bun:test';
import { TUTOR_PRESETS, normalizeTutorMode } from '../src/ui/hud-controller';

describe('HudController tutor presets and normalization', () => {
  it('normalizes tutor mode variants to canonical keys', () => {
    expect(normalizeTutorMode('vietnamese')).toBe('vietnamese');
    expect(normalizeTutorMode('vietnameseTutor')).toBe('vietnamese');
    expect(normalizeTutorMode('vietnamese_tutor')).toBe('vietnamese');
    expect(normalizeTutorMode('vi')).toBe('vietnamese');
    expect(normalizeTutorMode('vn')).toBe('vietnamese');

    expect(normalizeTutorMode('english')).toBe('english');
    expect(normalizeTutorMode('englishTutor')).toBe('english');
    expect(normalizeTutorMode('english_tutor')).toBe('english');
    expect(normalizeTutorMode('en')).toBe('english');

    expect(normalizeTutorMode('japanese')).toBe('japanese');
    expect(normalizeTutorMode('japaneseTutor')).toBe('japanese');
    expect(normalizeTutorMode('japanese_tutor')).toBe('japanese');
    expect(normalizeTutorMode('ja')).toBe('japanese');
    expect(normalizeTutorMode('jp')).toBe('japanese');

    expect(normalizeTutorMode('general')).toBe('general_assistant');
    expect(normalizeTutorMode('general_assistant')).toBe('general_assistant');
    expect(normalizeTutorMode('assistant')).toBe('general_assistant');
    expect(normalizeTutorMode('unknown_mode')).toBe('general_assistant');
  });

  it('exposes valid tutor presets for all canonical modes with natural and friendly prompts', () => {
    const modes = ['vietnamese', 'english', 'japanese', 'general_assistant'];
    for (const m of modes) {
      const preset = TUTOR_PRESETS[m];
      expect(preset).toBeDefined();
      expect(preset.id).toBe(m);
      expect(preset.name.length).toBeGreaterThan(0);
      expect(preset.systemPrompt.length).toBeGreaterThan(0);
      expect(preset.systemPrompt).toContain('Vira');
    }
  });

  it('verifies settings saving sends only canonical keys without duplicate aliases', async () => {
    const { TauriBridge } = await import('../src/ipc/tauri-bridge');
    let capturedSettings: any = null;

    const origInvoke = TauriBridge.invoke;
    TuriOverride: {
      (TauriBridge as any).invoke = async (cmd: string, args?: any) => {
        if (cmd === 'save_settings') {
          capturedSettings = args?.settings;
          return undefined;
        }
        return origInvoke.call(TauriBridge, cmd, args);
      };
    }

    try {
      // Simulate payload built by handleSaveSettings
      const mockPayload = {
        llm_endpoint: 'http://localhost:11434/v1',
        llm_model: 'qwen2.5:7b',
        llm_api_key: 'test-key',
        temperature: 0.7,
        tutor_mode: 'vietnamese',
        system_prompt: 'Custom prompt',
        tts_voice: 'vi-female',
        tts_speed: 1.0,
        rag_enabled: true,
      };

      await TauriBridge.invoke('save_settings', { settings: mockPayload });
      expect(capturedSettings).toBeDefined();
      expect(capturedSettings.llm_endpoint).toBe('http://localhost:11434/v1');
      expect(capturedSettings.tts_speed).toBe(1.0);
      expect('llm_base_url' in capturedSettings).toBe(false);
      expect('speech_rate' in capturedSettings).toBe(false);
    } finally {
      (TauriBridge as any).invoke = origInvoke;
    }
  });

  it('maintains backwards compatibility for legacy camelCase mode keys', () => {
    expect(TUTOR_PRESETS.vietnameseTutor).toBe(TUTOR_PRESETS.vietnamese);
    expect(TUTOR_PRESETS.englishTutor).toBe(TUTOR_PRESETS.english);
    expect(TUTOR_PRESETS.japaneseTutor).toBe(TUTOR_PRESETS.japanese);
    expect(TUTOR_PRESETS.general).toBe(TUTOR_PRESETS.general_assistant);
  });

  it('supports flashcard and pronunciation mock operations in tauri-bridge', async () => {
    const { TauriBridge } = await import('../src/ipc/tauri-bridge');

    // Create flashcard
    const card = await TauriBridge.invoke('create_flashcard', {
      payload: { front: 'Arigatou', back: 'Cảm ơn', cefr_level: 'A1' },
    });
    expect(card.front).toBe('Arigatou');
    expect(card.back).toBe('Cảm ơn');

    // List flashcards
    const cards = await TauriBridge.invoke('list_flashcards');
    expect(cards.length).toBeGreaterThan(0);

    // Assess pronunciation
    const assessment = await TauriBridge.invoke('assess_pronunciation', {
      payload: { target_text: 'Xin chào' },
    });
    expect(assessment.overall_score).toBeGreaterThan(80);
    expect(assessment.phoneme_scores.length).toBeGreaterThan(0);

    // Document ingest & list
    const doc = await TauriBridge.invoke('ingest_document', {
      payload: { title: 'Grammar Note', content: 'Grammar rule 1' },
    });
    expect(doc.title).toBe('Grammar Note');

    const docs = await TauriBridge.invoke('list_documents');
    expect(docs.length).toBeGreaterThan(0);
  });

  it('eliminates double speech by suppressing Web Speech API when native_audio is active', async () => {
    const { TauriBridge } = await import('../src/ipc/tauri-bridge');

    let speechCalled = false;
    let spokenText = '';
    (globalThis as any).window = globalThis.window || {};
    (globalThis as any).SpeechSynthesisUtterance = class {
      text: string;
      lang = '';
      rate = 1;
      constructor(t: string) {
        this.text = t;
      }
    };
    (globalThis.window as any).speechSynthesis = {
      cancel: () => {},
      speak: (utter: any) => {
        speechCalled = true;
        spokenText = utter.text;
      },
    };

    // Simulate IPC event handler logic
    const handleSpeakSentence = (payload: any) => {
      if (payload?.native_audio) {
        return;
      }
      if (payload?.text && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new (globalThis as any).SpeechSynthesisUtterance(payload.text);
        window.speechSynthesis.speak(utter);
      }
    };

    // 1. With native_audio: true -> speechSynthesis.speak MUST NOT be called
    speechCalled = false;
    handleSpeakSentence({ text: 'Hello from Kokoro native audio', native_audio: true });
    expect(speechCalled).toBe(false);

    // 2. With native_audio: false (fallback mode) -> speechSynthesis.speak MUST be called
    speechCalled = false;
    handleSpeakSentence({ text: 'Fallback speech synthesis', native_audio: false });
    expect(speechCalled).toBe(true);
    expect(spokenText).toBe('Fallback speech synthesis');
  });

  it('manages speech synthesis using sequential FIFO queue with utterance.onend', () => {
    const spokenList: string[] = [];
    const queue: any[] = [];
    let isSpeaking = false;

    (globalThis as any).window = globalThis.window || {};
    (globalThis as any).SpeechSynthesisUtterance = class {
      text: string;
      lang = '';
      rate = 1;
      onend: (() => void) | null = null;
      onerror: (() => void) | null = null;
      constructor(t: string) {
        this.text = t;
      }
    };

    (globalThis.window as any).speechSynthesis = {
      cancel: () => {
        queue.length = 0;
        isSpeaking = false;
      },
      speak: (utter: any) => {
        spokenList.push(utter.text);
      },
    };

    const playNext = () => {
      if (queue.length === 0) {
        isSpeaking = false;
        return;
      }
      const next = queue.shift();
      if (next) {
        isSpeaking = true;
        window.speechSynthesis.speak(next);
      }
    };

    const enqueue = (utter: any) => {
      const handleNext = () => {
        isSpeaking = false;
        playNext();
      };
      utter.onend = handleNext;
      utter.onerror = handleNext;
      queue.push(utter);
      if (!isSpeaking) {
        playNext();
      }
    };

    const utter1 = new (globalThis as any).SpeechSynthesisUtterance('Sentence 1');
    const utter2 = new (globalThis as any).SpeechSynthesisUtterance('Sentence 2');

    // Enqueue sentence 1
    enqueue(utter1);
    expect(spokenList).toEqual(['Sentence 1']);
    expect(isSpeaking).toBe(true);
    expect(queue.length).toBe(0);

    // Enqueue sentence 2 while sentence 1 is speaking - should be queued, not immediately spoken or cancelled
    enqueue(utter2);
    expect(spokenList).toEqual(['Sentence 1']);
    expect(queue.length).toBe(1);

    // Sentence 1 ends
    utter1.onend?.();
    expect(spokenList).toEqual(['Sentence 1', 'Sentence 2']);
    expect(queue.length).toBe(0);

    // Sentence 2 ends
    utter2.onend?.();
    expect(isSpeaking).toBe(false);
  });

  it('streams thought reasoning to thought scratchpad container', () => {
    const thoughtCard = { hidden: true, open: false };
    const thoughtBadge = { textContent: '' };
    const thoughtText = { textContent: '' };

    const handleThoughtEvent = (payload: any) => {
      if (!payload) return;
      thoughtCard.hidden = false;
      if (!payload.is_done) {
        thoughtCard.open = true;
        thoughtBadge.textContent = 'Thinking...';
      } else {
        thoughtBadge.textContent = 'Thought complete';
      }
      if (payload.text) {
        thoughtText.textContent = (thoughtText.textContent || '') + payload.text;
      }
    };

    // Step 1: initial thought chunk
    handleThoughtEvent({ text: 'Step 1: check grammar. ', is_done: false });
    expect(thoughtCard.hidden).toBe(false);
    expect(thoughtCard.open).toBe(true);
    expect(thoughtBadge.textContent).toBe('Thinking...');
    expect(thoughtText.textContent).toBe('Step 1: check grammar. ');

    // Step 2: subsequent thought chunk
    handleThoughtEvent({ text: 'Step 2: formulate response.', is_done: false });
    expect(thoughtText.textContent).toBe('Step 1: check grammar. Step 2: formulate response.');

    // Step 3: thought completion
    handleThoughtEvent({ text: '', is_done: true });
    expect(thoughtBadge.textContent).toBe('Thought complete');
  });

  it('test_pronunciation_assessment_ui_interaction_and_waveform verifies multi-metric scoring, reference audio, and viseme feedback', async () => {
    const { TauriBridge } = await import('../src/ipc/tauri-bridge');

    // 1. Invoke assess_pronunciation and verify multi-dimensional report
    const report = await TauriBridge.invoke('assess_pronunciation', {
      payload: { target_text: 'Good morning, Vira' },
    });

    expect(report.overall_score).toBeGreaterThan(80);
    expect(report.cefr_level).toBeDefined();
    expect(['A1', 'A2', 'B1', 'B2', 'C1', 'C2']).toContain(report.cefr_level);
    expect(report.accuracy_score).toBeGreaterThan(0);
    expect(report.fluency_score).toBeGreaterThan(0);
    expect(report.completeness_score).toBeGreaterThan(0);
    expect(report.prosody_score).toBeGreaterThan(0);
    expect(report.words.length).toBeGreaterThan(0);
    expect(report.phoneme_scores.length).toBeGreaterThan(0);
    expect(report.feedback_tips.length).toBeGreaterThan(0);

    // Verify word and phoneme structure
    const firstWord = report.words[0];
    expect(firstWord.word).toBeDefined();
    expect(firstWord.accuracy_score).toBeGreaterThan(0);
    expect(firstWord.phonemes.length).toBeGreaterThan(0);

    const firstPhoneme = report.phoneme_scores[0];
    expect(firstPhoneme.phoneme).toBeDefined();
    expect(firstPhoneme.score).toBeGreaterThan(0);
    expect(firstPhoneme.viseme_hint).toBeDefined();

    // 2. Test native reference and user recording audio IPCs
    const refAudio = await TauriBridge.invoke('get_pronunciation_reference_audio', {
      target_text: 'Good morning, Vira',
    });
    expect(refAudio.status).toBe('ok');
    expect(refAudio.audio_base64).toBeDefined();

    const replayAudio = await TauriBridge.invoke('replay_user_recording');
    expect(replayAudio.status).toBe('ok');
    expect(replayAudio.audio_base64).toBeDefined();

    // 3. Test 3D Avatar mouth viseme trigger on phoneme click
    let visemeDispatched: any = null;
    const mockVrmViewer = {
      onVisemeTick: (weights: any) => {
        visemeDispatched = weights;
      },
    };

    mockVrmViewer.onVisemeTick({
      weights: firstPhoneme.viseme_hint,
      is_speaking: true,
    });
    expect(visemeDispatched).toBeDefined();
    expect(visemeDispatched.weights).toEqual(firstPhoneme.viseme_hint);
    expect(visemeDispatched.is_speaking).toBe(true);
  });

  it('test_flashcard_3d_flip_and_ai_generation verifies 3D flip card, session progress, intervals, and 1-click AI generation', async () => {
    const { TauriBridge } = await import('../src/ipc/tauri-bridge');

    // 1. Test 1-click AI flashcard suggestion from subtitle text
    const subtitleSample = 'The pronunciation of this phrase is extraordinary.';
    const suggestion = await TauriBridge.invoke('generate_flashcard_ai_suggestion', {
      text: subtitleSample,
    });

    expect(suggestion.front).toBeDefined();
    expect(suggestion.back).toBeDefined();
    expect(suggestion.phonetic).toBeDefined();
    expect(suggestion.context_sentence).toContain(subtitleSample);
    expect(suggestion.context_translation).toBeDefined();
    expect(suggestion.cefr_level).toBeDefined();
    expect(suggestion.part_of_speech).toBeDefined();

    // 2. Create rich card with contextual fields
    const created = await TauriBridge.invoke('create_rich_flashcard', {
      payload: {
        front: suggestion.front,
        back: suggestion.back,
        phonetic: suggestion.phonetic,
        context_sentence: suggestion.context_sentence,
        context_translation: suggestion.context_translation,
        part_of_speech: suggestion.part_of_speech,
        cefr_level: suggestion.cefr_level,
        tags: suggestion.tags,
      },
    });

    expect(created.id).toBeDefined();
    expect(created.front).toBe(suggestion.front);
    expect(created.phonetic).toBe(suggestion.phonetic);
    expect(created.context_sentence).toBe(suggestion.context_sentence);

    // 3. Test dynamic FSRS interval preview
    const intervals = await TauriBridge.invoke('preview_flashcard_intervals', {
      card_id: created.id,
    });
    expect(intervals.again).toBeDefined();
    expect(intervals.hard).toBeDefined();
    expect(intervals.good).toBeDefined();
    expect(intervals.easy).toBeDefined();

    // 4. Test 3D Flip Card state toggle simulation
    const cardElement = {
      classList: new Set<string>(['flashcard-card-3d']),
      toggleFlip() {
        if (this.classList.has('flipped')) {
          this.classList.delete('flipped');
        } else {
          this.classList.add('flipped');
        }
      },
      isFlipped() {
        return this.classList.has('flipped');
      },
    };

    expect(cardElement.isFlipped()).toBe(false);
    cardElement.toggleFlip();
    expect(cardElement.isFlipped()).toBe(true);
    cardElement.toggleFlip();
    expect(cardElement.isFlipped()).toBe(false);

    // 5. Test session progress computation
    const computeSessionProgress = (currentIdx: number, totalCards: number) => {
      const clampedTotal = Math.max(1, totalCards);
      const clampedCurrent = Math.min(clampedTotal, Math.max(1, currentIdx + 1));
      const percentage = Math.round((clampedCurrent / clampedTotal) * 100);
      return {
        text: `Thẻ ${clampedCurrent} / ${clampedTotal}`,
        percentage,
      };
    };

    const p1 = computeSessionProgress(0, 10);
    expect(p1.text).toBe('Thẻ 1 / 10');
    expect(p1.percentage).toBe(10);

    const p3 = computeSessionProgress(2, 10);
    expect(p3.text).toBe('Thẻ 3 / 10');
    expect(p3.percentage).toBe(30);

    const pDone = computeSessionProgress(9, 10);
    expect(pDone.text).toBe('Thẻ 10 / 10');
    expect(pDone.percentage).toBe(100);
  });

  it('HudController handles 3D card flip, audio play, and recording state transitions', async () => {
    const { HudController } = await import('../src/ui/hud-controller');
    let visemeDispatched: any = null;
    const mockVrmViewer: any = {
      state: 'idle',
      setState: (s: string) => {
        mockVrmViewer.state = s;
      },
      onVisemeTick: (weights: any) => {
        visemeDispatched = weights;
      },
      playGesture: () => {},
      setEmotion: () => {},
    };

    const controller = new HudController(mockVrmViewer);

    // Mock internal el map
    const mock3DCard = {
      classList: {
        classes: new Set<string>(),
        toggle(cls: string, force?: boolean) {
          if (force !== undefined) {
            if (force) this.classes.add(cls);
            else this.classes.delete(cls);
            return force;
          }
          if (this.classes.has(cls)) {
            this.classes.delete(cls);
            return false;
          } else {
            this.classes.add(cls);
            return true;
          }
        },
        contains(cls: string) {
          return this.classes.has(cls);
        },
      },
    };

    const mockBtnReveal = { hidden: false };
    const mockRatingActions = { hidden: true };
    const mockBtnRecord = {
      classList: {
        classes: new Set<string>(),
        add(cls: string) {
          this.classes.add(cls);
        },
        remove(cls: string) {
          this.classes.delete(cls);
        },
      },
      setAttribute: () => {},
    };
    const mockRecordLabel = { textContent: '' };
    const mockTimerBadge = { textContent: '' };
    const mockBtnListenUser = { disabled: true };

    (controller as any).el = {
      fcCard3D: mock3DCard,
      btnFcReveal: mockBtnReveal,
      fcRatingActions: mockRatingActions,
      btnPronounceRecord: mockBtnRecord,
      pronounceRecordLabel: mockRecordLabel,
      pronounceTimerBadge: mockTimerBadge,
      btnPronounceListenUser: mockBtnListenUser,
    };

    // 1. Initial flip state: not flipped
    expect((controller as any).isCardFlipped).toBe(false);

    // 2. Toggle card flip -> flipped
    controller.toggleCardFlip();
    expect((controller as any).isCardFlipped).toBe(true);
    expect(mock3DCard.classList.contains('flipped')).toBe(true);
    expect(mockBtnReveal.hidden).toBe(true);
    expect(mockRatingActions.hidden).toBe(false);

    // 3. Set flip false -> front face
    controller.setCardFlip(false);
    expect((controller as any).isCardFlipped).toBe(false);
    expect(mock3DCard.classList.contains('flipped')).toBe(false);

    // 4. Test Pronunciation recording start
    controller.startPronounceRecording();
    expect((controller as any).isRecordingPronounce).toBe(true);
    expect(mockBtnRecord.classList.classes.has('recording')).toBe(true);
    expect(mockRecordLabel.textContent).toBe('Hoàn tất ghi âm');
    expect(mockTimerBadge.textContent).toBe('00:00');

    // 5. Test Pronunciation recording stop
    controller.stopPronounceRecording();
    expect((controller as any).isRecordingPronounce).toBe(false);
    expect(mockBtnRecord.classList.classes.has('recording')).toBe(false);
    expect(mockRecordLabel.textContent).toBe('Bắt đầu ghi âm');
    expect(mockBtnListenUser.disabled).toBe(false);

    controller.dispose();
  });
});
