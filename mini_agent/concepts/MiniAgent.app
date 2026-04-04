app MiniAgent
  purpose
    A production-quality AI coding agent that executes bounded think-act-observe
    cycles with tool use, permission sandboxing, context management, and
    cross-session memory consolidation.

  concepts
    Configuration
    MessageLog
    ContextBudget [Window := 200000, Threshold := 150000]
    ToolExecution [ToolRegistry := RegisteredTools]
    Permission [Mode := auto]
    AgentLoop [MaxSteps := 50]
    Session [SessionId := UUID]
    Memory [MemoryDir := ".claude/memory"]

  sync
    // Agent loop drives tool execution
    when AgentLoop.act ()
      then ToolExecution.request (tc, name, args)

    // Permission gates tool execution
    when ToolExecution.request (tc, name, args)
      then Permission.evaluate (name, args, new Decision)

    // Tool results feed back into the message log
    when ToolExecution.succeed (tc, content)
      then MessageLog.append_tool_result (new Message, content)
    when ToolExecution.fail (tc, error)
      then MessageLog.append_tool_result (new Message, error)

    // LLM response recorded in message log
    when AgentLoop.think ()
      then MessageLog.append_assistant (new Message, response)

    // Context budget tracks message growth
    when MessageLog.append_user (m, text)
      then ContextBudget.record_growth (estimate(text))
    when MessageLog.append_assistant (m, text)
      then ContextBudget.record_growth (estimate(text))
    when MessageLog.append_tool_result (m, output)
      then ContextBudget.record_growth (estimate(output))

    // Compaction when budget exceeded
    when ContextBudget.compact ()
      then MessageLog.replace_prefix (old_msgs, summary)

    // Session lifecycle brackets the agent loop
    when AgentLoop.begin (limit)
      then Session.start (session)
    when AgentLoop.complete ()
      then Session.end (session, "done")
    when AgentLoop.cancel ()
      then Session.end (session, "cancelled")
    when AgentLoop.exhaust ()
      then Session.end (session, "errored")

    // Memory consolidation after session ends
    when Session.end (s, reason)
      then Memory.dream (s.transcript)
           Memory.rebuild_index ()
