app MiniAgent
  purpose
    A production-quality AI coding agent that executes bounded think-act-observe
    cycles with tool use, permission sandboxing, context management, phase gating,
    codebase awareness, and cross-session memory consolidation.

  concepts
    Configuration
    RepoMap [ProjectDir := workspace, CharBudget := 6000]
    MessageLog
    ContextBudget [Window := 200000, ThresholdPct := 0.85, Reserve := 20000]
    ToolExecution [ToolRegistry := RegisteredTools]
    Permission [Mode := auto]
    AgentLoop [MaxSteps := 50]
    PhaseGating [ExploreRatio := 0.3, LastResortRatio := 0.8]
    Session [SessionId := UUID]
    Memory [MemoryDir := ".claude/memory"]

  sync
    // Repo map injected into system prompt on startup
    when Session.start (s)
      then RepoMap.get ()
           MessageLog.init (new Message, system_prompt + RepoMap.inject())

    // Agent loop drives tool execution
    when AgentLoop.act ()
      then ToolExecution.request (tc, name, args)

    // Permission gates tool execution using tool properties
    when ToolExecution.request (tc, name, args)
      then Permission.evaluate (name, args, is_read_only(tc), new Decision)

    // Tool results feed back into the message log
    when ToolExecution.succeed (tc, content)
      then MessageLog.append_tool_result (new Message, content)
    when ToolExecution.fail (tc, error)
      then MessageLog.append_tool_result (new Message, error)

    // LLM response recorded in message log
    when AgentLoop.think ()
      then MessageLog.append_assistant (new Message, response)

    // Context budget polls each step (not reactive on individual appends)
    when AgentLoop.act ()
      then ContextBudget.check (log)
    when ContextBudget.maybe_compact (log)
      then MessageLog.snip_observation (old_tool_msgs)
           MessageLog.replace_prefix (old_msgs, summary)

    // Phase gating monitors progress and injects guidance
    when AgentLoop.begin (limit)
      then PhaseGating.init (limit)
    when AgentLoop.act ()
      then PhaseGating.tick ()
           PhaseGating.check_explore_budget ()
           PhaseGating.check_last_resort ()
    when AgentLoop.complete ()
      then PhaseGating.check_verification (true)
    when ToolExecution.succeed (tc, content)
      where tool_name(tc) in ("edit_file", "write_file")
      then PhaseGating.record_edit ()
    when ToolExecution.succeed (tc, content)
      where tool_name(tc) = "bash" and content contains "test"
      then PhaseGating.record_test ()

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
      then Memory.dream (active, s.transcript)
           Memory.rebuild_index ()
