import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RouterProvider, createMemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { setApi } from "@/api";
import { createLiveApi } from "@/api/live";
import { createMockApi } from "@/api/mock/mockApi";
import { routes } from "@/router";
import { body, router as routeFetch } from "@/test/productApi";

function renderAt(path: string, options: Parameters<typeof createMockApi>[0] = {}) {
  setApi(createMockApi({ latencyMs: 0, ...options }));
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(<RouterProvider router={router} />);
  return router;
}

describe("screens on the mock adapter", () => {
  it("lists projects and always labels the data as mock", async () => {
    renderAt("/projects");
    const table = await screen.findByRole("table", { name: "Projects" }, { timeout: 3000 });
    expect(within(table).getByRole("link", { name: "Support triage" })).toHaveAttribute("href", "/projects/p-support");
    expect(within(table).getByRole("link", { name: "Invoice extraction" })).toBeInTheDocument();
    expect(screen.getAllByTestId("data-source")[0]).toHaveTextContent("Mock data");
  });

  it("shows an error state with retry when the adapter fails", async () => {
    renderAt("/projects", { failOn: ["listProjects"] });
    const alert = await screen.findByRole("alert", {}, { timeout: 3000 });
    expect(alert).toHaveTextContent("Could not load projects");
    expect(within(alert).getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("separates hard constraints, the optimization preference and the model configuration on the configure screen", async () => {
    renderAt("/projects/p-support/experiments/new?dataset=d-tickets");
    expect(await screen.findByText("Hard constraints", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText("Optimization preference")).toBeInTheDocument();
    expect(screen.getByText("Evaluation")).toBeInTheDocument();
    // models are the model configuration, not a hard constraint
    const models = screen.getByRole("heading", { name: "Models" }).closest("section")!;
    expect(within(models).getByRole("checkbox", { name: /Gemma 3 27B/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /^Classifications*Pick one label/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /^Classification accuracy/ })).toBeChecked();
    expect((screen.getByLabelText("Labels") as HTMLInputElement).value).toContain("billing");
    expect(screen.getByLabelText("Maximum cost ($ per 1k examples)")).toBeInTheDocument();
    expect(screen.getByLabelText("Maximum mean latency (s)")).toBeInTheDocument();

    const user = userEvent.setup();
    // minimizing cost needs a quality floor (core/task_contract.py)
    await user.click(screen.getByRole("radio", { name: /^Cost/ }));
    await user.type(screen.getByLabelText("Instructions"), "Route the ticket.");
    await user.click(screen.getByRole("button", { name: "Start optimization" }));
    expect(await screen.findByText(/Minimizing cost needs a minimum quality/)).toBeInTheDocument();
    // validation and test must leave rows for optimization
    const test = screen.getByLabelText("Held-out test (%)");
    await user.clear(test);
    await user.type(test, "75");
    expect(await screen.findByText("Leave at least 10% of rows for optimization.")).toBeInTheDocument();
  });

  it("asks for balanced weights and scales, and shows the utility formula", async () => {
    renderAt("/projects/p-support/experiments/new?dataset=d-tickets");
    const user = userEvent.setup();
    await user.click(await screen.findByRole("radio", { name: /^Balanced/ }, { timeout: 3000 }));
    expect(screen.getByLabelText("Quality weight")).toHaveValue(70);
    expect(
      screen.getByText("utility = 0.70 × quality − 0.20 × cost ÷ $0.500 / 1k − 0.10 × mean latency ÷ 2.0 s", { exact: false }),
    ).toBeInTheDocument();
  });

  it("compares the three methods on all splits for a finished run, with measured facts and no deployment", async () => {
    renderAt("/projects/p-support/experiments/e-triage-1/results");
    const table = await screen.findByRole("table", { name: "Method comparison" }, { timeout: 3000 });
    for (const name of ["Fixed baseline", "Random search", "Wynk (ACO)"]) expect(within(table).getByText(name)).toBeInTheDocument();
    expect(within(table).queryByText("locked")).not.toBeInTheDocument();
    const facts = screen.getByTestId("why-facts");
    expect(facts).toHaveTextContent("Held-out test accuracy");
    expect(facts).toHaveTextContent("Selected on the validation split");
    expect(screen.getByTestId("split-uses-test")).toHaveTextContent("Used for: reporting");
    expect(screen.getByTestId("split-uses-validation")).not.toHaveTextContent("search feedback");
    expect(facts).toHaveTextContent("for fixed baseline");
    expect(screen.getByRole("button", { name: /Deploy champion/ })).toBeDisabled();
    expect(screen.getByText("Deployment is coming soon")).toBeInTheDocument();
  });

  it("shows a failed run's reason", async () => {
    renderAt("/projects/p-invoices/experiments/e-invoices-1");
    expect(await screen.findByText("The run failed", {}, { timeout: 3000 })).toBeInTheDocument();
    expect(screen.getByText(/HTTP 503/)).toBeInTheDocument();
  });
});

describe("screens on the live adapter (real product API responses)", () => {
  const PID = body("create_project").project_id as string;
  const UID = body("upload_csv").upload_id as string;

  function renderLive(path: string, names: string[]) {
    const r = routeFetch(names);
    setApi(createLiveApi({ fetch: r.fetch }));
    render(<RouterProvider router={createMemoryRouter(routes, { initialEntries: [path] })} />);
    return r;
  }

  it("shows a dataset version, its hashes and its splits exactly as the server reported them", async () => {
    renderLive(`/projects/${PID}/datasets/support-tickets?version=1`, ["list_projects", "get_project", "get_dataset", "get_upload", "list_splits"]);
    const facts = await screen.findByTestId("version-facts", {}, { timeout: 3000 });
    expect(screen.getAllByTestId("data-source")[0]).toHaveTextContent("Live API");
    expect(within(facts).getByTestId("version-content-hash")).toHaveTextContent(body("register_v1").spec.content_hash);
    expect(within(facts).getByTestId("version-identity-hash")).toHaveTextContent(body("register_v1").identity_hash);
    expect(within(facts).getByTestId("version-rows")).toHaveTextContent("8");
    const row = await screen.findByTestId("splits-row");
    expect(row).toHaveAttribute("data-splits-hash", body("create_splits").splits_hash);
    const sizes = body("create_splits").sizes;
    expect(within(row).getByTestId("size-optimization")).toHaveTextContent(String(sizes.optimization));
    expect(within(row).getByTestId("size-validation")).toHaveTextContent(String(sizes.validation));
    expect(within(row).getByTestId("size-test")).toHaveTextContent(String(sizes.test));
    // experiments have no backend: the action says so instead of linking to a fake screen
    expect(screen.getByRole("button", { name: /Configure experiment · needs experiment backend/ })).toBeDisabled();
  });

  it("shows the server's inspection of an upload: types, nulls, row count, hash", async () => {
    renderLive(`/projects/${PID}/datasets/new?upload=${UID}`, ["list_projects", "get_project", "get_upload"]);
    expect(await screen.findByTestId("inspection-rows", {}, { timeout: 3000 })).toHaveTextContent("8");
    expect(screen.getByTestId("inspection-hash")).toHaveTextContent(body("upload_csv").content_hash);
    const schema = screen.getByRole("table", { name: "Columns and roles" });
    const tier = within(schema).getByText("customer_tier").closest("tr")!;
    expect(tier).toHaveTextContent("yes"); // nullable, as inferred by the server
    expect(tier).toHaveTextContent("1"); // one missing value over the whole file
    expect(screen.getByLabelText("Role of ticket_id")).toHaveValue("id");
  });

  it("surfaces a rejected upload with the server's reason and stable code", async () => {
    renderLive(`/projects/${PID}/datasets/new`, ["list_projects", "get_project", "error_malformed_csv"]);
    const input = await screen.findByTestId("dataset-file", {}, { timeout: 3000 });
    await userEvent.setup().upload(input, new File(["ticket_id,subject\nT-1\n"], "bad.csv", { type: "text/csv" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The CSV file is malformed");
    expect(alert).toHaveTextContent(body("error_malformed_csv").error.message);
    expect(within(alert).getByTestId("error-code")).toHaveTextContent("malformed_csv");
    expect(alert).toHaveTextContent("line 2");
  });

  it("says experiments need a backend that does not exist yet, and sends no experiment request", async () => {
    const r = renderLive(`/projects/${PID}/experiments`, ["list_projects", "get_project"]);
    expect(await screen.findByTestId("experiment-backend-required", {}, { timeout: 3000 })).toHaveTextContent(/not built yet/);
    for (const n of [20, 21, 22, 23]) expect(screen.getByRole("link", { name: `#${n}` })).toHaveAttribute("href", `https://github.com/Aryabhatta-0/Wynk/issues/${n}`);
    expect(r.calls.every((c) => !/experiment|workflow|model/.test(c))).toBe(true);
    expect(screen.queryByTestId("simulated-notice")).not.toBeInTheDocument();
  });
});

describe("experiment screens on the mock adapter", () => {
  it("label every simulated run as mock data on every experiment screen", async () => {
    for (const path of [
      "/projects/p-support/experiments",
      "/projects/p-support/experiments/e-triage-1",
      "/projects/p-support/experiments/e-triage-1/results",
      "/projects/p-support/workflows",
    ]) {
      const memory = renderAt(path);
      expect(await screen.findByTestId("simulated-notice", {}, { timeout: 3000 })).toHaveTextContent("No model was called and no optimization ran");
      memory.dispose();
      cleanup();
    }
  });
});
