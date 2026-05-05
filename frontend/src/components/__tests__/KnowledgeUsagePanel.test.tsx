import { act } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { KnowledgeUsagePanel } from '../SessionDetailSections'

import type { KnowledgeUsageEntry } from '@/types/api'

describe('KnowledgeUsagePanel', () => {
  let container: HTMLDivElement
  let root: Root

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
    root = createRoot(container)
  })

  afterEach(() => {
    act(() => {
      root.unmount()
    })
    container.remove()
  })

  it('renders grouped knowledge references once with usage count text', () => {
    const phases: Record<string, KnowledgeUsageEntry[]> = {
      planning: [
        {
          knowledge_item_id: 'item-1',
          title: 'Planning Output Format Guide',
          knowledge_type: 'format_guide',
          confidence_avg: 0.25,
          confidence_max: 0.3,
          retrieval_reason: 'sqlite_fallback_qdrant_or_embedding_unavailable',
          used_in_prompt: true,
          usage_count: 6,
          first_used_at: '2026-05-01T12:00:00Z',
          last_used_at: '2026-05-01T12:30:00Z',
        },
      ],
    }

    act(() => {
      root.render(<KnowledgeUsagePanel phases={phases} />)
    })

    expect(container.textContent).toContain('Planning Output Format Guide')
    expect(container.textContent).toContain('30%')
    expect(container.textContent).toContain('injected')
    expect(container.textContent).toContain('used 6 times')
    expect(container.textContent).toContain(
      'format_guide • sqlite_fallback_qdrant_or_embedding_unavailable'
    )
    expect(container.textContent?.match(/Planning Output Format Guide/g)?.length).toBe(1)
  })

  it('shows retrieved label for non-injected references', () => {
    const phases: Record<string, KnowledgeUsageEntry[]> = {
      failure: [
        {
          knowledge_item_id: 'item-2',
          title: 'Failure Case',
          knowledge_type: 'debug_case',
          confidence_avg: 0.8,
          confidence_max: 0.9,
          retrieval_reason: 'semantic_retrieval',
          used_in_prompt: false,
          usage_count: 1,
          first_used_at: '2026-05-01T12:00:00Z',
          last_used_at: '2026-05-01T12:00:00Z',
        },
      ],
    }

    act(() => {
      root.render(<KnowledgeUsagePanel phases={phases} />)
    })

    expect(container.textContent).toContain('retrieved')
    expect(container.textContent).not.toContain('used 1 times')
  })
})
