import { useState, useEffect } from 'react';
import { Pencil, Plus, Settings, Trash2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { useGatewayStore } from '../store/useGatewayStore';
import { useT } from '../i18n/useT';
import * as api from '../api/client';
import {
  gatewayActivateModel,
  gatewayCreateModel,
  gatewayDeleteModel,
  gatewayGetModelStages,
  gatewayListModels,
  gatewaySetModelStages,
  gatewayUpdateModel,
  type GatewayModelInfo,
} from '../api/gateway';

export function SettingsModal() {
  const {
    theme, fontSize, agentMapBackground,
    setTheme, setFontSize, setAgentMapBackground,
    settingsPanelOpen, setSettingsPanelOpen,
  } = useAppStore();
  const t = useT();
  const { token, sessionId, testConnection } = useGatewayStore();

  const [backgrounds, setBackgrounds] = useState<string[]>([]);

  const [llmModel, setLlmModel] = useState(t.settingsModal.deepseekV4Flash);
  const [models, setModels] = useState<GatewayModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState('');
  const [selectedCoderId, setSelectedCoderId] = useState('');
  const [selectedFeedbackId, setSelectedFeedbackId] = useState('');
  const [modelForm, setModelForm] = useState({ model_name: '', api_key: '', api_url: '' });
  const [modelsLoading, setModelsLoading] = useState(false);
  const [addingModel, setAddingModel] = useState(false);
  const [activatingModel, setActivatingModel] = useState(false);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  const [modelManagerOpen, setModelManagerOpen] = useState(false);
  const [editingModelId, setEditingModelId] = useState('');
  const [deletingModel, setDeletingModel] = useState(false);
  const [status, setStatus] = useState(t.settingsModal.ready);
  const [saving, setSaving] = useState(false);

  // Agent Map backgrounds come from <program dir>/themes/backgrounds; the
  // option label is the image file name.
  useEffect(() => {
    api.fetchThemeBackgrounds()
      .then(setBackgrounds)
      .catch(() => setBackgrounds([]));
  }, []);

  useEffect(() => {
    if (!token) {
      setModels([]);
      setSelectedModelId('');
      setSelectedCoderId('');
      setSelectedFeedbackId('');
      return;
    }
    setModelsLoading(true);
    gatewayListModels(token)
      .then((items) => {
        setModels(items);
        const active = items.find((item) => item.is_active) || items[0];
        setSelectedModelId(active?.id || '');
        if (active) setLlmModel(active.model_name);
        if (sessionId) {
          return gatewayGetModelStages(token, sessionId).then((stages) => {
            // Stage dropdowns display the main model's entry unless an
            // explicit override was saved for the session.
            setSelectedCoderId(stages.coder_model_id || active?.id || '');
            setSelectedFeedbackId(stages.feedback_model_id || active?.id || '');
          });
        }
      })
      .catch((e: unknown) => setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`))
      .finally(() => setModelsLoading(false));
  }, [token, sessionId, t.settingsModal.unknownError]);

  const handleModelChange = async (modelId: string) => {
    setSelectedModelId(modelId);
    // Stage dropdowns follow the main model unless they were explicitly
    // overridden (i.e. their current value differs from the old main model).
    setSelectedCoderId((cur) => (cur === selectedModelId ? modelId : cur));
    setSelectedFeedbackId((cur) => (cur === selectedModelId ? modelId : cur));
    const model = models.find((item) => item.id === modelId);
    if (!model || !token || !sessionId) return;
    setActivatingModel(true);
    setStatus(t.settingsModal.activating);
    try {
      const active = await gatewayActivateModel(token, modelId, sessionId);
      setModels((items) => items.map((item) => ({ ...item, is_active: item.id === active.id })));
      setLlmModel(active.model_name);
      setStatus(t.settingsModal.modelActivated);
    } catch (e: unknown) {
      setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    } finally {
      setActivatingModel(false);
    }
  };

  const handleStageChange = async (coderId: string, feedbackId: string) => {
    if (!token || !sessionId) return;
    setActivatingModel(true);
    try {
      const stages = await gatewaySetModelStages(token, sessionId, coderId, feedbackId);
      setSelectedCoderId(stages.coder_model_id || '');
      setSelectedFeedbackId(stages.feedback_model_id || '');
      setStatus(t.settingsModal.saved);
    } catch (e: unknown) {
      setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    } finally {
      setActivatingModel(false);
    }
  };

  const handleAddModel = async () => {
    if (!token) return;
    setAddingModel(true);
    setStatus(editingModelId ? t.settingsModal.updatingModel : t.settingsModal.addingModel);
    try {
      if (editingModelId) {
        await gatewayUpdateModel(token, editingModelId, modelForm);
        const items = await gatewayListModels(token);
        setModels(items);
        const updated = items.find((item) => item.id === editingModelId);
        if (updated && selectedModelId === updated.id) setLlmModel(updated.model_name);
        setStatus(t.settingsModal.modelUpdated);
      } else {
        await gatewayCreateModel(token, modelForm);
        const items = await gatewayListModels(token);
        setModels(items);
        const active = items.find((item) => item.is_active) || items[0];
        if (!selectedModelId && active) {
          setSelectedModelId(active.id);
          setLlmModel(active.model_name);
        }
        setStatus(t.settingsModal.modelAdded);
      }
      setModelForm({ model_name: '', api_key: '', api_url: '' });
      setEditingModelId('');
      setModelDialogOpen(false);
    } catch (e: unknown) {
      setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    } finally {
      setAddingModel(false);
    }
  };

  const handleEditModel = (model: GatewayModelInfo) => {
    setEditingModelId(model.id);
    setModelForm({ model_name: model.model_name, api_key: model.api_key, api_url: model.api_url });
    setModelDialogOpen(true);
  };

  const handleOpenAddModel = () => {
    setEditingModelId('');
    setModelForm({ model_name: '', api_key: '', api_url: '' });
    setModelDialogOpen(true);
  };

  const handleDeleteModel = async (modelId = selectedModelId) => {
    if (!token || !modelId || deletingModel) return;
    if (!window.confirm(t.settingsModal.confirmDeleteModel)) return;
    setDeletingModel(true);
    try {
      await gatewayDeleteModel(token, modelId);
      const items = await gatewayListModels(token);
      setModels(items);
      const active = items.find((item) => item.is_active) || items[0];
      setSelectedModelId(active?.id || '');
      setLlmModel(active?.model_name || '');
      setStatus(t.settingsModal.modelDeleted);
    } catch (e: unknown) {
      setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    } finally {
      setDeletingModel(false);
    }
  };

  // Gateway-only connectivity check: probes the log gateway itself and
  // updates the shared status shown in the topbar.
  const handleTest = async () => {
    setStatus(t.settingsModal.testing);
    try {
      const health = await testConnection();
      setStatus(`${t.settingsModal.connectionOk} · ${health.latency_ms}ms · ${health.url}`);
    } catch (e: unknown) {
      setStatus(`${t.settingsModal.connectionFailed}: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setStatus(t.settingsModal.saving);
    try {
      if (token && sessionId && selectedModelId) {
        const active = await gatewayActivateModel(token, selectedModelId, sessionId);
        setModels((items) => items.map((item) => ({ ...item, is_active: item.id === active.id })));
        setLlmModel(active.model_name);
      }
      setStatus(t.settingsModal.saved);
    } catch (e: unknown) {
      setStatus(`Failed: ${(e as Error)?.message || t.settingsModal.unknownError}`);
    } finally {
      setSaving(false);
    }
  };

  const handleClose = () => {
    setModelDialogOpen(false);
    setModelManagerOpen(false);
    setSettingsPanelOpen(false);
  };

  const handleResetModelForm = () => {
    setModelForm({ model_name: '', api_key: '', api_url: '' });
  };

  const handleCloseModelDialog = () => {
    if (!addingModel) {
      setModelDialogOpen(false);
      setEditingModelId('');
    }
  };

  return (
    <div className="settings-modal" data-settings-modal>
      <div className="settings-scrim" data-settings-close onClick={handleClose}></div>
      <section className="settings-card" role="dialog" aria-modal="true">
        <div className="card-head">
          <span className="card-title">{t.settingsModal.settings}</span>
          <button className="btn" type="button" data-settings-close onClick={handleClose}>
            {t.settingsModal.done}
          </button>
        </div>
        <div className="settings-body">
          <div className="settings-section">
            <div className="settings-section-title">{t.settingsModal.display}</div>
            <div className="field">
              <label>{t.settingsModal.theme}</label>
              <select
                data-settings-theme
                value={theme}
                onChange={(e) => setTheme(e.target.value as typeof theme)}
              >
                <option value="scienceflow-dark">{t.settingsModal.dark}</option>
                <option value="paper-light">{t.settingsModal.paperLight}</option>
              </select>
            </div>
            <div className="field">
              <label>{t.settingsModal.textSize}</label>
              <select
                data-settings-font-size
                value={fontSize}
                onChange={(e) => setFontSize(e.target.value as typeof fontSize)}
              >
                <option value="small">{t.settingsModal.small}</option>
                <option value="default">{t.settingsModal.default}</option>
                <option value="large">{t.settingsModal.large}</option>
              </select>
            </div>
            <div className="field">
              <label>{t.settingsModal.backgroundImage}</label>
              <select
                data-settings-background-image
                value={agentMapBackground}
                onChange={(e) => setAgentMapBackground(e.target.value)}
              >
                <option value="">{t.settingsModal.agentMapBackground}</option>
                {(agentMapBackground && !backgrounds.includes(agentMapBackground)
                  ? [agentMapBackground, ...backgrounds]
                  : backgrounds
                ).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="settings-section">
            <div className="settings-section-title settings-model-title">
              <span>{t.settingsModal.model}</span>
              <div className="settings-model-tools">
                <button
                  className="btn icon-btn"
                  type="button"
                  onClick={handleOpenAddModel}
                  disabled={!token || addingModel || deletingModel}
                  aria-label={t.settingsModal.addModel}
                  title={t.settingsModal.addModel}
                >
                  <Plus size={16} />
                </button>
                <button
                  className="btn icon-btn"
                  type="button"
                  onClick={() => setModelManagerOpen(true)}
                  disabled={!token || modelsLoading}
                  aria-label={t.settingsModal.modelManagement}
                  title={t.settingsModal.modelManagement}
                >
                  <Settings size={16} />
                </button>
              </div>
            </div>
            <div className="settings-model-row">
              <div className="field">
                <label>{t.settingsModal.model}</label>
                <select
                  data-settings-llm-model
                  value={selectedModelId}
                  onChange={(e) => void handleModelChange(e.target.value)}
                  disabled={modelsLoading || activatingModel || models.length === 0}
                >
                  {models.length === 0 ? (
                    <option value="">{modelsLoading ? t.settingsModal.loadingModels : t.settingsModal.noModels}</option>
                  ) : models.map((item) => (
                    <option key={item.id} value={item.id}>{item.model_name}</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="settings-model-row">
              <div className="field">
                <label>{t.settingsModal.coderModel}</label>
                <select
                  value={selectedCoderId}
                  onChange={(e) => void handleStageChange(e.target.value, selectedFeedbackId)}
                  disabled={modelsLoading || activatingModel || models.length === 0}
                >
                  {models.map((item) => (
                    <option key={item.id} value={item.id}>{item.model_name}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>{t.settingsModal.feedbackModel}</label>
                <select
                  value={selectedFeedbackId}
                  onChange={(e) => void handleStageChange(selectedCoderId, e.target.value)}
                  disabled={modelsLoading || activatingModel || models.length === 0}
                >
                  {models.map((item) => (
                    <option key={item.id} value={item.id}>{item.model_name}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <div className="settings-status" data-settings-status>{status}</div>
        </div>

        <div className="settings-actions">
          <button
            className="btn"
            type="button"
            data-settings-test
            onClick={handleTest}
          >
            {t.settingsModal.test}
          </button>
          <button
            className="btn primary"
            type="button"
            data-settings-save
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? t.settingsModal.saving : t.settingsModal.save}
          </button>
        </div>
      </section>

      {modelManagerOpen && (
        <div className="settings-scrim add-model-scrim" onClick={() => setModelManagerOpen(false)}></div>
      )}

      {modelManagerOpen && (
        <section className="settings-card add-model-dialog model-manager-dialog" role="dialog" aria-modal="true">
          <div className="card-head add-model-head">
            <span className="card-title">{t.settingsModal.modelManagement}</span>
            <button className="btn icon-btn" type="button" onClick={() => setModelManagerOpen(false)} aria-label={t.settingsModal.close}>
              ×
            </button>
          </div>
          <div className="settings-body model-manager-body">
            <div className="model-manager-toolbar">
              <button className="btn" type="button" onClick={handleOpenAddModel} disabled={!token || addingModel || deletingModel}>
                <Plus size={14} />
                <span>{t.settingsModal.addModel}</span>
              </button>
            </div>
            <div className="model-manager-table">
              <div className="model-manager-head">
                <span>{t.settingsModal.model}</span>
                <span>{t.settingsModal.provider}</span>
                <span>{t.settingsModal.operation}</span>
              </div>
              <div className="model-provider-group">
                {models.length === 0 ? (
                  <div className="model-empty-row">{modelsLoading ? t.settingsModal.loadingModels : t.settingsModal.noModels}</div>
                ) : models.map((model) => (
                  <div className="model-manager-row" key={model.id}>
                    <div className="model-name-cell">
                      <span className="model-cube-icon">◈</span>
                      <span>{model.model_name}</span>
                    </div>
                    <div className="model-provider-cell" title={model.api_url}>{model.api_url}</div>
                    <div className="model-operation-cell">
                      <button className="btn icon-btn ghost-icon" type="button" onClick={() => handleEditModel(model)} disabled={addingModel || deletingModel} aria-label={t.settingsModal.editModel} title={t.settingsModal.editModel}>
                        <Pencil size={12} />
                      </button>
                      <button className="btn icon-btn ghost-icon danger" type="button" onClick={() => void handleDeleteModel(model.id)} disabled={addingModel || deletingModel} aria-label={t.settingsModal.deleteModel} title={t.settingsModal.deleteModel}>
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      {modelDialogOpen && (
        <div className="settings-scrim add-model-scrim" onClick={handleCloseModelDialog}></div>
      )}

      {modelDialogOpen && (
        <section className="settings-card add-model-dialog model-form-dialog" role="dialog" aria-modal="true">
          <div className="card-head add-model-head">
            <span className="card-title">{editingModelId ? t.settingsModal.editModel : t.settingsModal.addModel}</span>
            <button className="btn icon-btn" type="button" onClick={handleCloseModelDialog} aria-label={t.settingsModal.close} disabled={addingModel}>
              ×
            </button>
          </div>
          <div className="settings-body add-model-body">
            <div className="settings-grid add-model-fields">
              <div className="field">
                <label>{t.settingsModal.modelName}<span className="required-star">*</span></label>
                <input required value={modelForm.model_name} placeholder={t.settingsModal.modelNamePlaceholder} onChange={(e) => setModelForm({ ...modelForm, model_name: e.target.value })} />
              </div>
              <div className="field">
                <label>{t.settingsModal.apiKey}<span className="required-star">*</span></label>
                <input required type="password" value={modelForm.api_key} placeholder={t.settingsModal.apiKeyPlaceholder} onChange={(e) => setModelForm({ ...modelForm, api_key: e.target.value })} />
              </div>
              <div className="field">
                <label>{t.settingsModal.apiUrl}<span className="required-star">*</span></label>
                <input required value={modelForm.api_url} placeholder={t.settingsModal.apiUrlPlaceholder} onChange={(e) => setModelForm({ ...modelForm, api_url: e.target.value })} />
              </div>
            </div>
          </div>
          <div className="settings-actions">
            <button className="btn" type="button" onClick={handleResetModelForm} disabled={addingModel}>
              {t.settingsModal.reset}
            </button>
            <button className="btn primary" type="button" onClick={() => void handleAddModel()} disabled={addingModel || !token || !modelForm.model_name || !modelForm.api_key || !modelForm.api_url}>
              {addingModel ? (editingModelId ? t.settingsModal.updatingModel : t.settingsModal.addingModel) : (editingModelId ? t.settingsModal.updateModel : t.settingsModal.addModel)}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
