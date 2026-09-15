// Backend + checkpoint pickers. The logic and persistence live in
// client/shared/inference.ts, shared with every world — see
// ../../kanban/components/InferenceControls.tsx for why the markup itself
// still isn't shared even in one React tree.
import { INFERENCE_MODES, type InferenceMode, type ModelInfo } from '../../../../../shared/inference';

interface Props {
  mode: InferenceMode;
  model: string | null;
  models: ModelInfo[];
  disabled: boolean;
  onMode: (mode: InferenceMode) => void;
  onModel: (model: string) => void;
}

export function InferenceControls({ mode, model, models, disabled, onMode, onModel }: Props) {
  const current = models.find((m) => m.name === model);
  return (
    <>
      <label className="picker" title={INFERENCE_MODES.find((m) => m.mode === mode)?.hint ?? ''}>
        Inference
        <select value={mode} disabled={disabled} onChange={(e) => onMode(e.target.value as InferenceMode)}>
          {INFERENCE_MODES.map((m) => (
            <option key={m.mode} value={m.mode}>
              {m.label.toLowerCase()}
            </option>
          ))}
        </select>
      </label>
      {models.length > 0 && (
        <label
          className="picker"
          title={
            current
              ? `${current.size_mb} MB · ${current.tuned ? 'tuned on our IR corpus' : 'not tuned; prompted with its own chat template'}`
              : 'checkpoint served from baselines/qwen/models'
          }
        >
          Model
          <select value={model ?? ''} disabled={disabled} onChange={(e) => onModel(e.target.value)}>
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name.replace(/\.gguf$/, '')}
              </option>
            ))}
          </select>
        </label>
      )}
    </>
  );
}
