import { TavilySearch } from "@langchain/tavily";
import { ChatOpenAI } from "@langchain/openai";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import dotenv from "dotenv";

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

	const finalState = await app.invoke({ messages: [new HumanMessage("when and where is the ipl final match 2025")] });

	console.log(finalState.messages[finalState.messages.length - 1].content);

	const nextState = await app.invoke({
		messages: [...finalState.messages, new HumanMessage("what are the chances of rain for that match")],
	});

	console.log(nextState.messages[nextState.messages.length - 1].content);
}

agent().catch(console.error);
