import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import dotenv from "dotenv";
import readlineSync from "readline-sync";

dotenv.config();

const openAIApiKey = process.env.OPENAI_API_KEY;
const tavilyApiKey = process.env.TAVILY_API_KEY;

async function agent() {
	const tools = [new TavilySearch({ maxResults: 3, tavilyApiKey })];
	const toolNode = new ToolNode(tools);

	const model = new ChatOpenAI({ temperature: 0, openAIApiKey, modelName: "gpt-4o" }).bindTools(tools);

	function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
		const lastMessage = messages[messages.length - 1] as AIMessage;

		if (lastMessage.tool_calls?.length) {
			console.log("[LOG] Detected tool call. Routing to tools node.");
			return "tools";
		}

		return "__end__";
	}

	async function callModel(state: typeof MessagesAnnotation.State) {
		const response = await model.invoke(state.messages);
		return { messages: [response] };
	}

	const workflow = new StateGraph(MessagesAnnotation)
		.addNode("agent", callModel)
		.addEdge("__start__", "agent")
		.addNode("tools", toolNode)
		.addEdge("tools", "agent")
		.addConditionalEdges("agent", shouldContinue);

	const app = workflow.compile();

	let messages: (HumanMessage | AIMessage)[] = [];

	console.log("Welcome to the ReAct Agent. Type 'exit' to end the conversation.\n");

	while (true) {
		const input = readlineSync.question("> You: ");
		if (input.trim().toLowerCase() === "exit") break;

		messages.push(new HumanMessage(input));
		const state = await app.invoke({ messages });
		messages = state.messages;

		const lastMessage = messages[messages.length - 1] as AIMessage;
		console.log(`AI: ${lastMessage.content}\n`);
	}
}

agent().catch(console.error);
