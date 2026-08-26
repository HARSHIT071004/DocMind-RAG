import { AgentChat, type AgentMessage } from "./agent-chat";

const messages: AgentMessage[] = [
  {
    id: "msg-1",
    role: "user",
    parts: [{ type: "text", text: "Show me the latest status." }],
  },
  {
    id: "msg-2",
    role: "assistant",
    parts: [{ type: "text", text: "All systems are green." }],
  },
];

export default function BasicDemo() {
  return (
    <div className="w-full h-screen p-4 bg-neutral-50 dark:bg-neutral-950">
      <div className="w-full max-w-[680px] h-full mx-auto rounded-xl border border-neutral-200 dark:border-neutral-800 overflow-hidden bg-white dark:bg-neutral-900">
        <AgentChat
          messages={messages}
          status="ready"
          onSend={() => {}}
          onStop={() => {}}
        />
      </div>
    </div>
  );
}
