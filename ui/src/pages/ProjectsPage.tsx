import { FolderSimplePlus, Plus } from "@phosphor-icons/react";
import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { api, errorMessage } from "@/api";
import { useShell } from "@/components/app/AppShell";
import { FlowSteps } from "@/components/app/FlowSteps";
import { EmptyState, ErrorState, Field, LoadingState, PageHeader, Panel } from "@/components/app/ui";
import { date, int } from "@/lib/format";
import { useResource } from "@/lib/useResource";

export function ProjectsPage() {
  const projects = useResource(() => api().listProjects(), []);
  const [creating, setCreating] = useState(false);

  return (
    <div className="space-y-5">
      <PageHeader
        kicker={<FlowSteps current="Project" />}
        title="Projects"
        description="A project groups the datasets for one task, the optimization experiments run on them, and the workflows they produce."
        actions={
          !creating && (
            <button type="button" className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
              <Plus size={14} weight="bold" aria-hidden="true" /> New project
            </button>
          )
        }
      />

      {creating && <NewProjectForm onCancel={() => setCreating(false)} />}

      <Panel bodyClassName="p-0">
        {projects.state === "loading" && (
          <div className="p-4">
            <LoadingState label="Loading projects" />
          </div>
        )}
        {projects.state === "error" && (
          <div className="p-4">
            <ErrorState title="Could not load projects" message={projects.error} onRetry={projects.reload} />
          </div>
        )}
        {projects.state === "ready" &&
          (projects.data.length === 0 ? (
            <EmptyState
              icon={FolderSimplePlus}
              title="No projects yet"
              action={
                <button type="button" className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
                  <Plus size={14} weight="bold" aria-hidden="true" /> New project
                </button>
              }
            >
              Create a project for one task, such as routing support tickets, then add a dataset of examples.
            </EmptyState>
          ) : (
            <table className="data-table" aria-label="Projects">
              <thead>
                <tr>
                  <th>Project</th>
                  <th className="num">Datasets</th>
                  <th className="num">Experiments</th>
                  <th className="num">Created</th>
                </tr>
              </thead>
              <tbody>
                {projects.data.map((p) => (
                  <tr key={p.id} className="hover:bg-wash/60">
                    <td>
                      <Link to={`/projects/${p.id}`} className="font-semibold text-ink hover:text-magenta-ink">
                        {p.name}
                      </Link>
                      {p.description && <p className="mt-0.5 max-w-[80ch] truncate text-xs text-ink-soft">{p.description}</p>}
                    </td>
                    <td className="num">{int(p.datasetCount)}</td>
                    <td className="num">{int(p.experimentCount)}</td>
                    <td className="num text-ink-soft">{date(p.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ))}
      </Panel>
    </div>
  );
}

function NewProjectForm({ onCancel }: { onCancel: () => void }) {
  const navigate = useNavigate();
  const { refreshProjects } = useShell();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return setError("Name the project.");
    setSaving(true);
    setError(null);
    try {
      const p = await api().createProject({ name, description });
      refreshProjects();
      navigate(`/projects/${p.id}/datasets`);
    } catch (err) {
      setError(errorMessage(err));
      setSaving(false);
    }
  };

  return (
    <Panel title="New project">
      <form onSubmit={submit} className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)_auto] md:items-start" noValidate>
        <Field label="Name" error={error ?? undefined}>
          {(p) => (
            <input
              {...p}
              className="input"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Support triage"
              autoFocus
              maxLength={80}
            />
          )}
        </Field>
        <Field label="Description" hint="Optional. What the workflows in this project should do.">
          {(p) => <input {...p} className="input" value={description} onChange={(e) => setDescription(e.target.value)} maxLength={200} />}
        </Field>
        <div className="flex gap-2 md:pt-6">
          <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
            {saving ? "Creating…" : "Create project"}
          </button>
          <button type="button" className="btn btn-quiet btn-sm" onClick={onCancel} disabled={saving}>
            Cancel
          </button>
        </div>
      </form>
    </Panel>
  );
}
