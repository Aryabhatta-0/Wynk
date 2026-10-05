/*
  MOCK ONLY. Seed data for the mock adapter. Names, rows, hashes and numbers are invented for the
  UI; they are not benchmark results. Every dataset and configuration is a valid instance of the
  merged contracts (DatasetSpec roles and types, TaskContract task/evaluator rules, ObjectiveSpec,
  ConstraintLimits units), and the mock builds the TaskContract from them to prove it.
*/
import type { Dataset, ExperimentConfig, ModelOption, Project } from "../types";
import { hex } from "./random";

export const MODELS: ModelOption[] = [
  { id: "gemma-3-12b-it", label: "Gemma 3 12B", provider: "Google" },
  { id: "gemma-3-27b-it", label: "Gemma 3 27B", provider: "Google" },
  { id: "gemma-4-31b-it", label: "Gemma 4 31B", provider: "Google" },
  { id: "llama-3.3-70b-instruct", label: "Llama 3.3 70B", provider: "Meta" },
];

const DAY = 86_400_000;
const ago = (days: number) => new Date(Date.now() - days * DAY).toISOString();
const fakeHash = (id: string) => hex(`mock-content:${id}`, 64);

export const PROJECTS: Project[] = [
  {
    id: "p-support",
    name: "Support triage",
    description: "Route inbound tickets to the right queue and answer common questions.",
    createdAt: ago(12),
    datasetCount: 0,
    experimentCount: 0,
  },
  {
    id: "p-invoices",
    name: "Invoice extraction",
    description: "Pull vendor, totals and due dates out of supplier invoices.",
    createdAt: ago(5),
    datasetCount: 0,
    experimentCount: 0,
  },
];

export const DATASETS: Dataset[] = [
  {
    id: "d-tickets",
    projectId: "p-support",
    name: "Support tickets, Q3",
    format: "csv",
    fileName: "support_tickets_q3.csv",
    sizeBytes: 1_284_096,
    contentHash: fakeHash("d-tickets"),
    version: 1,
    rowCount: 2400,
    createdAt: ago(11),
    status: "ready",
    columns: [
      { name: "ticket_id", type: "string", nullable: false, missing: 0, distinct: 6 },
      { name: "subject", type: "string", nullable: false, missing: 0, distinct: 6 },
      { name: "body", type: "string", nullable: false, missing: 0, distinct: 6 },
      { name: "customer_tier", type: "string", nullable: true, missing: 1, distinct: 3 },
      { name: "queue", type: "string", nullable: false, missing: 0, distinct: 4 },
    ],
    preview: [
      {
        ticket_id: "T-10422",
        subject: "Charged twice this month",
        body: "My card shows two charges for the Pro plan on the 3rd.",
        customer_tier: "pro",
        queue: "billing",
      },
      {
        ticket_id: "T-10423",
        subject: "API returns 502",
        body: "Since this morning every call to /v2/orders fails with 502.",
        customer_tier: "enterprise",
        queue: "technical",
      },
      {
        ticket_id: "T-10424",
        subject: "Change account owner",
        body: "Our admin left; please move ownership to dana@acme.test.",
        customer_tier: "pro",
        queue: "account",
      },
      {
        ticket_id: "T-10425",
        subject: "Parcel stuck in transit",
        body: "Tracking has not moved for six days, order 88213.",
        customer_tier: "",
        queue: "shipping",
      },
      {
        ticket_id: "T-10426",
        subject: "Refund for annual plan",
        body: "We downgraded mid-cycle and expected a prorated refund.",
        customer_tier: "free",
        queue: "billing",
      },
      {
        ticket_id: "T-10427",
        subject: "SSO login loop",
        body: "Okta redirects back to the login page endlessly.",
        customer_tier: "enterprise",
        queue: "technical",
      },
    ],
    mapping: { input: ["subject", "body"], target: ["queue"], context: ["customer_tier"], id: "ticket_id" },
  },
  {
    id: "d-faq",
    projectId: "p-support",
    name: "Help-centre questions",
    format: "jsonl",
    fileName: "faq_eval.jsonl",
    sizeBytes: 902_144,
    contentHash: fakeHash("d-faq"),
    version: 1,
    rowCount: 1500,
    createdAt: ago(3),
    status: "needs_mapping",
    columns: [
      { name: "faq_id", type: "integer", nullable: false, missing: 0, distinct: 4 },
      { name: "question", type: "string", nullable: false, missing: 0, distinct: 4 },
      { name: "passage", type: "string", nullable: false, missing: 0, distinct: 4 },
      { name: "answer", type: "string", nullable: false, missing: 0, distinct: 4 },
    ],
    preview: [
      {
        faq_id: 101,
        question: "How long do refunds take?",
        passage: "Refunds are issued to the original payment method within 5-7 business days.",
        answer: "5-7 business days",
      },
      {
        faq_id: 102,
        question: "Can I change my billing date?",
        passage: "Billing dates can be moved once per year from Settings > Billing.",
        answer: "Yes, once per year",
      },
      { faq_id: 103, question: "Is there a free plan?", passage: "The Free plan includes 3 projects and community support.", answer: "Yes" },
      {
        faq_id: 104,
        question: "Where do I find invoices?",
        passage: "Invoices are listed under Settings > Billing > History.",
        answer: "Settings > Billing > History",
      },
    ],
    // not mapped yet: no target and no id column
    mapping: { input: ["question"], target: [], context: [], id: null },
  },
  {
    id: "d-invoices",
    projectId: "p-invoices",
    name: "Supplier invoices",
    format: "jsonl",
    fileName: "invoices.jsonl",
    sizeBytes: 3_420_160,
    contentHash: fakeHash("d-invoices"),
    version: 1,
    rowCount: 860,
    createdAt: ago(4),
    status: "ready",
    columns: [
      { name: "doc_id", type: "string", nullable: false, missing: 0, distinct: 3 },
      { name: "document_text", type: "string", nullable: false, missing: 0, distinct: 3 },
      { name: "vendor_country", type: "string", nullable: false, missing: 0, distinct: 3 },
      { name: "vendor", type: "string", nullable: false, missing: 0, distinct: 3 },
      { name: "total", type: "number", nullable: false, missing: 0, distinct: 3 },
      { name: "currency", type: "string", nullable: false, missing: 0, distinct: 2 },
      { name: "due_date", type: "date", nullable: false, missing: 0, distinct: 3 },
    ],
    preview: [
      {
        doc_id: "INV-2207",
        document_text: "Nordlicht GmbH · Invoice 2207 · Total EUR 1,240.00 · Due 2026-11-02",
        vendor_country: "DE",
        vendor: "Nordlicht GmbH",
        total: 1240,
        currency: "EUR",
        due_date: "2026-11-02",
      },
      {
        doc_id: "INV-2208",
        document_text: "Pacific Paper Co. — Amount due USD 318.50 by 10/28/2026",
        vendor_country: "US",
        vendor: "Pacific Paper Co.",
        total: 318.5,
        currency: "USD",
        due_date: "2026-10-28",
      },
      {
        doc_id: "INV-2209",
        document_text: "Atelier Rive · Facture 2209 · Montant TTC 96,00 € · Échéance 15/11/2026",
        vendor_country: "FR",
        vendor: "Atelier Rive",
        total: 96,
        currency: "EUR",
        due_date: "2026-11-15",
      },
    ],
    mapping: { input: ["document_text"], target: ["vendor", "total", "currency", "due_date"], context: ["vendor_country"], id: "doc_id" },
  },
];

export interface SeedExperiment {
  id: string;
  projectId: string;
  config: ExperimentConfig;
  createdDaysAgo: number;
  /** where the run stopped, 0..1; 1 = finished */
  stoppedAt: number;
  failure?: string;
}

export const EXPERIMENTS: SeedExperiment[] = [
  {
    id: "e-triage-1",
    projectId: "p-support",
    createdDaysAgo: 9,
    stoppedAt: 1,
    config: {
      name: "Ticket routing, quality first",
      datasetId: "d-tickets",
      taskType: "classification",
      instructions: "Read the ticket's subject and body and answer with the queue that should handle it.",
      evaluation: { evaluator: "classification_accuracy", labels: ["billing", "technical", "account", "shipping"], caseSensitive: false },
      constraints: {
        minQuality: 0.82,
        maxCostPerExample: 0.0006,
        maxMeanLatencyS: null,
        maxP95LatencyS: 9,
        maxTokensPerExample: null,
        maxWorkflowSteps: null,
      },
      preferences: { objective: "quality", balanced: null },
      models: ["gemma-3-27b-it", "gemma-4-31b-it"],
      budget: { maxCandidates: 96, maxGenerations: 12, maxSpendUsd: 40, maxDurationMin: 240 },
      splits: { validationPct: 20, testPct: 20, seed: 7 },
    },
  },
  {
    id: "e-invoices-1",
    projectId: "p-invoices",
    createdDaysAgo: 2,
    stoppedAt: 0.38,
    failure: "The model endpoint for gemma-4-31b-it returned HTTP 503 for 3 consecutive generations, so the run stopped.",
    config: {
      name: "Invoice fields under a cost cap",
      datasetId: "d-invoices",
      taskType: "structured_extraction",
      instructions: "Extract the vendor name, invoice total, currency code and due date (YYYY-MM-DD) from the invoice text.",
      evaluation: { evaluator: "exact_match", caseSensitive: false, normalizeWhitespace: true },
      constraints: {
        minQuality: 0.8,
        maxCostPerExample: 0.00025,
        maxMeanLatencyS: null,
        maxP95LatencyS: null,
        maxTokensPerExample: null,
        maxWorkflowSteps: null,
      },
      preferences: { objective: "cost", balanced: null },
      models: ["gemma-3-12b-it", "gemma-4-31b-it"],
      budget: { maxCandidates: 64, maxGenerations: 10, maxSpendUsd: 15, maxDurationMin: 60 },
      splits: { validationPct: 20, testPct: 20, seed: 11 },
    },
  },
];
