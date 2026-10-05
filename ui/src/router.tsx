import { Compass } from "@phosphor-icons/react";
import { lazy, Suspense } from "react";
import { Link, Navigate, type RouteObject, useRouteError } from "react-router";
import { AppShell } from "@/components/app/AppShell";
import { EmptyState, ErrorState } from "@/components/app/ui";
import { ConfigurePage } from "@/pages/ConfigurePage";
import { DatasetDetailPage } from "@/pages/DatasetDetailPage";
import { DatasetImportPage } from "@/pages/DatasetImportPage";
import { DatasetsPage } from "@/pages/DatasetsPage";
import { ExperimentPage } from "@/pages/ExperimentPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { ProjectLayout } from "@/pages/ProjectLayout";
import { ProjectsPage } from "@/pages/ProjectsPage";
import { ResultsPage } from "@/pages/ResultsPage";
import { WorkflowsPage } from "@/pages/WorkflowsPage";

// the hackathon chat demo, kept whole; loaded only when visited
const Playground = lazy(() => import("./Playground"));

/*
  Project → Dataset → Configure → Optimize → Compare → Champion

  /projects
  /projects/:projectId/datasets            list
  /projects/:projectId/datasets/new        file, preview, schema, column roles
  /projects/:projectId/datasets/:id        mapping and preview
  /projects/:projectId/experiments         list
  /projects/:projectId/experiments/new     configure
  /projects/:projectId/experiments/:id     run: progress, curve, candidates, search
  /projects/:projectId/experiments/:id/results   baseline vs random vs Wynk, champion, why
  /projects/:projectId/workflows           validated workflows and champions
  /playground                              ask-a-question chat demo
*/
export const routes: RouteObject[] = [
  {
    path: "/playground",
    element: (
      <Suspense fallback={null}>
        <Playground />
      </Suspense>
    ),
  },
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <Navigate to="/projects" replace /> },
      { path: "projects", element: <ProjectsPage /> },
      {
        path: "projects/:projectId",
        element: <ProjectLayout />,
        children: [
          { index: true, element: <Navigate to="datasets" replace /> },
          { path: "datasets", element: <DatasetsPage /> },
          { path: "datasets/new", element: <DatasetImportPage /> },
          { path: "datasets/:datasetId", element: <DatasetDetailPage /> },
          { path: "experiments", element: <ExperimentsPage /> },
          { path: "experiments/new", element: <ConfigurePage /> },
          { path: "experiments/:experimentId", element: <ExperimentPage /> },
          { path: "experiments/:experimentId/results", element: <ResultsPage /> },
          { path: "workflows", element: <WorkflowsPage /> },
        ],
      },
      { path: "*", element: <NotFound /> },
    ],
  },
];

function NotFound() {
  return (
    <EmptyState
      icon={Compass}
      title="Nothing here"
      action={
        <Link to="/projects" className="btn btn-primary btn-sm">
          Go to projects
        </Link>
      }
    >
      This address does not match a page.
    </EmptyState>
  );
}

function RouteError() {
  const error = useRouteError();
  return (
    <div className="mx-auto max-w-2xl p-8">
      <ErrorState title="Something broke on this page" message={error instanceof Error ? error.message : "An unexpected error occurred."} />
    </div>
  );
}
