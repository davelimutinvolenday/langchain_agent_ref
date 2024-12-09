import dotenv from 'dotenv';
import { Annotation } from '@langchain/langgraph';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { ChatOpenAI } from '@langchain/openai';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage } from '@langchain/core/messages';

dotenv.config();

const run = async () => {
	try {
		const tools = [new TavilySearchResults({ maxResults: 3 })];

		const agentExecutor = createReactAgent({
			llm: new ChatOpenAI({ modelName: 'gpt-4o' }),
			tools
		});

		const result = await agentExecutor.invoke({
			messages: [new HumanMessage('Who is the winner of the US open?')]
		});

		console.log('%c Line:35 🥕 result', 'color:#7f2b82', result);
	} catch (error) {
		console.log('%c Line:34 🍪 error', 'color:#42b983', error);
	}
};

run();
