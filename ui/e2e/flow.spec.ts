import { expect, test } from "@playwright/test";

const CSV = `ticket_id,subject,body,customer_tier,queue
T-1,Charged twice,"My card shows two charges, both for March.",pro,billing
T-2,Cannot log in,SSO redirects back to the login page.,enterprise,technical
T-3,Update company name,We rebranded; please change the account name.,pro,account
T-4,Where is my order,Tracking has not moved for a week.,free,shipping
T-5,Refund request,"Downgraded mid-cycle, expected a prorated refund.",pro,billing
T-6,Webhook failures,Deliveries to our endpoint time out since Monday.,enterprise,technical
T-7,Add a teammate,How do I invite a colleague to our workspace?,free,account
T-8,Damaged parcel,The box arrived crushed and the device is cracked.,pro,shipping
`;

test("mocked flow: project → dataset → configure → optimize → results", async ({ page }) => {
  // mock adapter knobs: a short run and small response delays
  await page.goto("/projects?mockRunMs=5000&mockLatency=40");
  await expect(page.getByTestId("data-source").first()).toContainText("Mock data");

  // Project
  await page.getByRole("button", { name: "New project" }).click();
  await page.getByLabel("Name").fill("E2E ticket routing");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name: "E2E ticket routing" })).toBeVisible();
  await expect(page.getByText("No datasets in this project")).toBeVisible();

  // Dataset: parsed in the browser, schema and suggested roles shown, then confirmed
  await page.getByRole("link", { name: "Add dataset" }).click();
  await page.getByTestId("dataset-file").setInputFiles({ name: "tickets.csv", mimeType: "text/csv", buffer: Buffer.from(CSV) });
  await expect(page.getByText(/^tickets\.csv · \d+ B · 8 rows · 5 columns$/)).toBeVisible();
  await expect(page.getByLabel("Role of queue")).toHaveValue("target");
  await expect(page.getByLabel("Role of subject")).toHaveValue("input");
  await expect(page.getByLabel("Role of customer_tier")).toHaveValue("context");
  await expect(page.getByLabel("Role of ticket_id")).toHaveValue("id");
  await expect(page.getByRole("table", { name: "Preview rows" }).getByRole("row")).toHaveCount(9);
  // a broken mapping blocks registration
  await page.getByLabel("Role of queue").selectOption("input");
  await expect(page.getByText("Choose at least one target column for the workflow to produce.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Register and configure" })).toBeDisabled();
  await page.getByLabel("Role of queue").selectOption("target");
  await page.getByRole("button", { name: "Register and configure" }).click();

  // Configure: task, evaluation, hard constraints, preference, models, budget
  await expect(page).toHaveURL(/\/experiments\/new\?dataset=/);
  await expect(page.getByRole("radio", { name: /^Classification\s*Pick one label/ })).toBeChecked();
  await expect(page.getByRole("radio", { name: /^Classification accuracy/ })).toBeChecked();
  await expect(page.getByLabel("Labels")).toHaveValue(/billing/);
  await expect(page.getByText("Hard constraints")).toBeVisible();
  await page.getByLabel("Instructions").fill("Read the ticket and answer with the queue that should handle it.");
  await page.getByLabel("Minimum quality (%)").fill("75");
  await page.getByLabel("Maximum p95 latency (s)").fill("10");
  await page.getByRole("checkbox", { name: /Gemma 4 31B/ }).check();
  await page.getByRole("radio", { name: /^Balanced/ }).check();
  await expect(page.getByText(/utility = 0\.70 × quality/)).toBeVisible();
  await page.getByLabel("Candidates", { exact: true }).fill("40");
  await page.getByLabel("Generations", { exact: true }).fill("5");
  await page.getByRole("button", { name: "Start optimization" }).click();

  // Optimize: live progress, then a champion
  const experiment = page.getByTestId("experiment");
  await expect(experiment).toHaveAttribute("data-status", /queued|running/);
  await expect(page.getByText("Candidates evaluated")).toBeVisible();
  await expect(experiment).toHaveAttribute("data-status", "completed", { timeout: 30_000 });
  await expect(page.getByRole("heading", { name: "Champion workflow" })).toBeVisible();
  expect(await page.getByTestId("candidate-row").count()).toBeGreaterThan(0);
  await page.getByRole("button", { name: /Search \(ACO\)/ }).click();
  await expect(page.getByRole("list", { name: "Most reinforced transitions" }).getByRole("listitem").first()).toBeVisible();

  // Compare: three methods, held-out test unlocked, measured facts, no deployment
  await page.getByRole("link", { name: "Compare results" }).click();
  await expect(page.getByTestId("results")).toBeVisible();
  for (const m of ["fixed_baseline", "random_search", "wynk_aco"]) await expect(page.getByTestId(`method-${m}`)).toBeVisible();
  await expect(page.getByRole("table", { name: "Method comparison" }).getByText("locked")).toHaveCount(0);
  await expect(page.getByTestId("why-facts")).toContainText("Held-out test accuracy");
  // the champion was chosen on validation; held-out test may confirm or break a constraint, and either is reported
  await expect(page.getByTestId("why-facts")).toContainText(/(Met all \d+ hard constraints|Broke [a-z0-9 ,]+) on the held-out test split/);
  await expect(page.getByRole("button", { name: /Deploy champion/ })).toBeDisabled();

  // Champion: listed with its state in the project's workflows
  await page.getByRole("link", { name: "Workflows" }).first().click();
  await expect(page.getByRole("table", { name: "Workflows" }).getByText("Champion")).toBeVisible();
});

test("error and live-mode states are explicit, never a fake success", async ({ page }) => {
  await page.goto("/projects?mockFail=listProjects&mockLatency=0");
  await expect(page.getByRole("alert").filter({ hasText: "Could not load projects" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();

  await page.goto("/projects?api=live");
  await expect(page.getByTestId("data-source").first()).toContainText("Live API");
  await expect(page.getByRole("alert").filter({ hasText: "not available yet" })).toBeVisible();
});
