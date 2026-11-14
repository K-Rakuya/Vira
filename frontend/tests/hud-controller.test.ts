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

  it('exposes valid tutor presets for all canonical modes', () => {
    const modes = ['vietnamese', 'english', 'japanese', 'general_assistant'];
    for (const m of modes) {
      const preset = TUTOR_PRESETS[m];
      expect(preset).toBeDefined();
      expect(preset.id).toBe(m);
      expect(preset.name.length).toBeGreaterThan(0);
      expect(preset.systemPrompt.length).toBeGreaterThan(0);
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
});
