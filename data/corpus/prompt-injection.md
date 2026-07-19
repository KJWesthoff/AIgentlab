# Prompt injection in tool-using LLM systems

Prompt injection occurs when untrusted content — a retrieved document, a
web page, an email, a tool result — contains text that a language model
interprets as instructions rather than data. In agentic systems this is a
critical risk because a successful injection can cause the model to
propose unauthorized tool calls, exfiltrate data, or ignore its task.

Mitigations include: labeling all retrieved content as untrusted data,
enforcing tool authorization in code outside the model (a policy
enforcement point), granting each agent only the minimum capabilities it
needs, requiring human approval for write-capable or high-impact actions,
and logging every proposed and executed action for audit.

The model proposes; the runtime decides. A model output is never
authorization. Systems that treat model text as commands without an
independent policy check are vulnerable by construction.
