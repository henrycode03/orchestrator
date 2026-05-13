import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

const FRONTEND_URL = process.env.PHASE9E_FRONTEND_URL || 'http://127.0.0.1:3000';
const TEST_EMAIL = process.env.PHASE9E_EMAIL || 'phase9-sweep@example.com';
const TEST_PASSWORD = process.env.PHASE9E_PASSWORD || 'Phase9SweepPass!2026';
const RUN_SESSION_LIFECYCLE = process.env.PHASE9E_RUN_SESSION_LIFECYCLE === '1';

async function ensureSignedIn(page: Page) {
  let loginResponse = await page.request.post(`${FRONTEND_URL}/api/v1/auth/session/login`, {
    data: { email: TEST_EMAIL, password: TEST_PASSWORD },
  });
  for (let attempt = 0; loginResponse.status() === 429 && attempt < 3; attempt += 1) {
    await page.waitForTimeout(15_000);
    loginResponse = await page.request.post(`${FRONTEND_URL}/api/v1/auth/session/login`, {
      data: { email: TEST_EMAIL, password: TEST_PASSWORD },
    });
  }
  expect(loginResponse.ok(), `login status ${loginResponse.status()}`).toBeTruthy();
  await page.goto(`${FRONTEND_URL}/dashboard`);
  await page.waitForLoadState('networkidle');
  if (await page.getByRole('heading', { name: 'Sign in' }).isVisible()) {
    await page.getByPlaceholder('you@example.com').fill(TEST_EMAIL);
    await page.getByPlaceholder('••••••••').fill(TEST_PASSWORD);
    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect(page.getByRole('heading', { name: 'Sign in' })).toBeHidden({ timeout: 30_000 });
  }
}

async function cleanupProject(page: Page, projectName: string) {
  const projectsResponse = await page.request.get(`${FRONTEND_URL}/api/v1/projects`);
  if (!projectsResponse.ok()) return;
  const projects = (await projectsResponse.json()) as Array<{ id: number; name: string }>;
  await Promise.all(
    projects
      .filter((project) => project.name === projectName || project.name.startsWith(projectName))
      .map((project) => page.request.delete(`${FRONTEND_URL}/api/v1/projects/${project.id}`))
  );
}

async function projectIdByName(page: Page, projectName: string) {
  const projectsResponse = await page.request.get(`${FRONTEND_URL}/api/v1/projects`);
  expect(projectsResponse.ok()).toBeTruthy();
  const projects = (await projectsResponse.json()) as Array<{ id: number; name: string }>;
  return projects.find((project) => project.name === projectName)?.id;
}

test.describe('Phase 9E disposable mutation sweep', () => {
  test('project mutation controls and session creation work with disposable data', async ({ page }) => {
    test.setTimeout(180_000);
    await page.context().clearCookies();
    await ensureSignedIn(page);

    const suffix = Date.now();
    const projectName = `phase9e disposable ${suffix}`;
    const renamedProjectName = `${projectName} renamed`;
    const sessionName = `phase9e session ${suffix}`;

    page.on('dialog', async (dialog) => {
      await dialog.accept();
    });

    await cleanupProject(page, 'phase9e disposable');
    await cleanupProject(page, projectName);
    await cleanupProject(page, renamedProjectName);

    await page.goto(`${FRONTEND_URL}/projects`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('button', { name: 'New Project' }).click();
    await page.getByPlaceholder('My Project').fill(projectName);
    await page
      .getByPlaceholder('What this project is for, scope, expected deliverable...')
      .fill('Disposable Phase 9E mutation sweep project.');
    await page
      .getByPlaceholder('Constraints, style rules, forbidden tools, must-keep architecture...')
      .fill('Created by automated UI mutation sweep; safe to delete.');

    const createProjectResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/v1/projects') &&
        response.request().method() === 'POST'
    );
    await page.getByRole('button', { name: 'Create' }).click();
    const createdProjectResponse = await createProjectResponse;
    await expect(createdProjectResponse.ok()).toBeTruthy();
    const createdProject = (await createdProjectResponse.json()) as { id: number; name: string };
    expect(createdProject.name).toBe(projectName);
    await page.goto(`${FRONTEND_URL}/projects`);
    await page.waitForLoadState('networkidle');
    await page.getByPlaceholder('Search...').fill(projectName);
    await expect(page.getByText(projectName, { exact: true })).toBeVisible();

    await page.getByTitle('Rename project').click();
    await page.locator('input[value="' + projectName + '"]').last().fill(renamedProjectName);
    const updateProjectResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/v1/projects/') &&
        response.request().method() === 'PUT'
    );
    await page.getByTitle('Save changes').click();
    await expect((await updateProjectResponse).ok()).toBeTruthy();
    await page.getByPlaceholder('Search...').fill(renamedProjectName);
    await expect(page.getByText(renamedProjectName, { exact: true })).toBeVisible();

    const projectId = await projectIdByName(page, renamedProjectName);
    expect(projectId, 'created project id').toBeTruthy();

    await page.goto(`${FRONTEND_URL}/sessions/new?project_id=${projectId}`);
    await page.waitForLoadState('networkidle');
    await page.getByPlaceholder('e.g., Vite Website Development').fill(sessionName);
    await page
      .getByPlaceholder('Describe what you want the AI session to accomplish...')
      .fill('Disposable Phase 9E session creation sweep.');
    await page.getByRole('button', { name: 'Manual' }).click();
    const createSessionResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/v1/sessions') &&
        response.request().method() === 'POST'
    );
    await page.getByRole('button', { name: /Create Session/i }).click();
    await expect((await createSessionResponse).ok()).toBeTruthy();
    await page.waitForURL(/\/sessions\/\d+/, { timeout: 60_000 });
    await expect(page.getByRole('heading', { name: sessionName })).toBeVisible();

    if (RUN_SESSION_LIFECYCLE) {
      await page.getByRole('button', { name: 'Start' }).click();
      await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeVisible({
        timeout: 60_000,
      });
      const sessionId = page.url().match(/\/sessions\/(\d+)/)?.[1];
      expect(sessionId, 'created session id').toBeTruthy();
      const interventionResponse = await page.request.post(
        `${FRONTEND_URL}/api/v1/sessions/${sessionId}/request-intervention`,
        {
          data: {
            intervention_type: 'guidance',
            initiated_by: 'ai',
            prompt: 'Phase 9E synthetic intervention: confirm the reply flow is usable.',
            context_snapshot: { source: 'phase9e_mutation_sweep' },
            expires_in_minutes: 15,
          },
        }
      );
      expect(interventionResponse.ok(), `intervention status ${interventionResponse.status()}`).toBeTruthy();
      await page.reload();
      await page.waitForLoadState('networkidle');
      await expect(page.getByText('OpenClaw Needs Your Input')).toBeVisible({
        timeout: 30_000,
      });
      await page
        .getByPlaceholder('Your guidance reply...')
        .first()
        .fill('Phase 9E synthetic operator reply.');
      await page.getByRole('button', { name: 'Submit Reply' }).first().click();
      await expect(page.getByText('OpenClaw Needs Your Input')).toBeHidden({
        timeout: 30_000,
      });
      await expect(page.getByText('Paused')).toBeVisible({ timeout: 30_000 });
      await page.getByRole('button', { name: /^Stop$/ }).click();
      await expect(page.getByRole('button', { name: 'Start' })).toBeVisible({
        timeout: 60_000,
      });
    }

    await page.goto(`${FRONTEND_URL}/projects`);
    await page.waitForLoadState('networkidle');
    await page.getByPlaceholder('Search...').fill(renamedProjectName);
    await page.getByTitle('Delete project').click();
    await expect(page.getByText(renamedProjectName, { exact: true })).toBeHidden({
      timeout: 30_000,
    });
    await cleanupProject(page, renamedProjectName);
  });
});
