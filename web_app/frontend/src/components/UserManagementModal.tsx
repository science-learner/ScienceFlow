import { useCallback, useEffect, useState } from 'react';
import { Pencil, Trash2 } from 'lucide-react';
import { useT } from '../i18n/useT';
import {
  createUser,
  deleteUser,
  listUsers,
  updateUser,
  type GatewayUserInfo,
  type UpdateUserPayload,
} from '../api/users';

interface UserManagementModalProps {
  open: boolean;
  onClose: () => void;
}

interface UserFormState {
  name: string;
  password: string;
  role: string;
}

const EMPTY_FORM: UserFormState = {
  name: '',
  password: '',
  role: 'user',
};

// Super-admin account management (see doc/stage2/user_admin.md): list, create,
// edit (password/role) and delete gateway accounts. Log-source permissions are
// intentionally not managed here (edit leaves them unchanged server-side;
// new accounts default to all sources). Rendered from the topbar and only
// reachable for role="admin" users; the gateway enforces the permission too.
export function UserManagementModal({ open, onClose }: UserManagementModalProps) {
  const t = useT();
  const [users, setUsers] = useState<GatewayUserInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [formOpen, setFormOpen] = useState(false);
  const [editingName, setEditingName] = useState('');
  const [form, setForm] = useState<UserFormState>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [deletingName, setDeletingName] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    setStatus('');
    try {
      setUsers(await listUsers());
    } catch (e) {
      setStatus(`${t.userAdmin.failed}: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [t.userAdmin.failed]);

  useEffect(() => {
    if (!open) return;
    void refresh();
  }, [open, refresh]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const openAdd = () => {
    setEditingName('');
    setForm(EMPTY_FORM);
    setFormOpen(true);
    setStatus('');
  };

  const openEdit = (user: GatewayUserInfo) => {
    setEditingName(user.name);
    setForm({
      name: user.name,
      password: '',
      role: user.admin ? 'admin' : 'user',
    });
    setFormOpen(true);
    setStatus('');
  };

  const closeForm = () => {
    if (saving) return;
    setFormOpen(false);
    setEditingName('');
  };

  const canSubmit = editingName
    ? true
    : form.name.trim().length > 0 && form.password.length > 0;

  const submitForm = async () => {
    if (!canSubmit || saving) return;
    setSaving(true);
    setStatus('');
    try {
      if (editingName) {
        // Partial update: only password (when provided) and role are sent, so
        // the account's existing source permissions stay untouched.
        const payload: UpdateUserPayload = { role: form.role };
        if (form.password) payload.password = form.password;
        const updated = await updateUser(editingName, payload);
        setUsers((prev) => prev.map((u) => (u.name === updated.name ? updated : u)));
        setStatus(t.userAdmin.updated);
      } else {
        const created = await createUser({
          name: form.name.trim(),
          password: form.password,
          role: form.role,
        });
        setUsers((prev) =>
          [...prev.filter((u) => u.name !== created.name), created].sort((a, b) =>
            a.name.localeCompare(b.name),
          ),
        );
        setStatus(t.userAdmin.created);
      }
      setFormOpen(false);
      setEditingName('');
      setForm(EMPTY_FORM);
    } catch (e) {
      setStatus(`${t.userAdmin.failed}: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (user: GatewayUserInfo) => {
    if (deletingName) return;
    if (!window.confirm(`${t.userAdmin.confirmDelete} (${user.name})`)) return;
    setDeletingName(user.name);
    setStatus('');
    try {
      await deleteUser(user.name);
      setUsers((prev) => prev.filter((u) => u.name !== user.name));
      setStatus(t.userAdmin.deleted);
    } catch (e) {
      setStatus(`${t.userAdmin.failed}: ${(e as Error).message}`);
    } finally {
      setDeletingName('');
    }
  };

  return (
    <div className="settings-modal">
      <div className="settings-scrim" data-user-admin-close onClick={onClose} />
      <section className="settings-card user-admin-card" role="dialog" aria-modal="true">
        <div className="card-head">
          <span className="card-title">{t.userAdmin.title}</span>
          <button className="btn" type="button" data-user-admin-close onClick={onClose}>
            {t.userAdmin.close}
          </button>
        </div>
        <div className="settings-body">
          <div className="model-manager-toolbar">
            <button
              className="btn"
              type="button"
              onClick={openAdd}
              disabled={loading || saving || !!deletingName}
            >
              {t.userAdmin.addUser}
            </button>
          </div>
          <div className="model-manager-table" data-user-admin-table>
            <div className="model-manager-head user-admin-row">
              <span>{t.userAdmin.name}</span>
              <span>{t.settingsModal.operation}</span>
            </div>
            <div className="model-provider-group">
              {loading ? (
                <div className="model-empty-row">{t.userAdmin.loading}</div>
              ) : users.length === 0 ? (
                <div className="model-empty-row" data-user-admin-empty>{t.userAdmin.empty}</div>
              ) : (
                users.map((user) => (
                  <div className="model-manager-row user-admin-row" key={user.name} data-user-admin-row={user.name}>
                    <div className="model-name-cell">
                      <span className="model-cube-icon">◈</span>
                      <span>{user.name}</span>
                      <span className={user.admin ? 'user-admin-badge' : 'user-admin-badge user'}>
                        {user.admin ? t.userAdmin.roleAdmin : t.userAdmin.roleUser}
                      </span>
                    </div>
                    <div className="model-operation-cell">
                      <button
                        className="btn icon-btn ghost-icon"
                        type="button"
                        onClick={() => openEdit(user)}
                        disabled={saving || !!deletingName}
                        aria-label={t.userAdmin.edit}
                        title={t.userAdmin.edit}
                      >
                        <Pencil size={12} />
                      </button>
                      <button
                        className="btn icon-btn ghost-icon danger"
                        type="button"
                        onClick={() => void handleDelete(user)}
                        disabled={saving || !!deletingName}
                        aria-label={t.userAdmin.delete}
                        title={t.userAdmin.delete}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
          <div className="settings-status" data-user-admin-status>{status}</div>
          <div className="dim" style={{ fontSize: 11 }}>{t.userAdmin.lastAdminHint}</div>
        </div>
      </section>

      {formOpen && (
        <div className="settings-scrim add-model-scrim" onClick={closeForm} />
      )}

      {formOpen && (
        <section className="settings-card add-model-dialog user-admin-dialog" role="dialog" aria-modal="true">
          <div className="card-head add-model-head">
            <span className="card-title">{editingName ? t.userAdmin.editUser : t.userAdmin.addUser}</span>
            <button
              className="btn icon-btn"
              type="button"
              onClick={closeForm}
              aria-label={t.userAdmin.close}
              disabled={saving}
            >
              ×
            </button>
          </div>
          <div className="settings-body add-model-body">
            <div className="settings-grid add-model-fields">
              <div className="field">
                <label>
                  {t.userAdmin.name}
                  <span className="required-star">*</span>
                </label>
                <input
                  value={form.name}
                  disabled={!!editingName}
                  placeholder={t.userAdmin.namePlaceholder}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                />
              </div>
              <div className="field">
                <label>
                  {t.userAdmin.password}
                  {!editingName && <span className="required-star">*</span>}
                </label>
                <input
                  type="password"
                  value={form.password}
                  placeholder={editingName ? t.userAdmin.passwordEditPlaceholder : t.userAdmin.passwordPlaceholder}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                />
              </div>
              <div className="field">
                <label>{t.userAdmin.role}</label>
                <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                  <option value="user">{t.userAdmin.roleUser}</option>
                  <option value="admin">{t.userAdmin.roleAdmin}</option>
                </select>
              </div>
            </div>
          </div>
          <div className="settings-actions">
            <button className="btn" type="button" onClick={closeForm} disabled={saving}>
              {t.userAdmin.cancel}
            </button>
            <button className="btn primary" type="button" onClick={() => void submitForm()} disabled={saving || !canSubmit}>
              {saving ? t.settingsModal.saving : editingName ? t.userAdmin.save : t.userAdmin.add}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}
