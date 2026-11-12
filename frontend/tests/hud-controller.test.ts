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
});
