import dotenv from 'dotenv';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { ChatOpenAI } from '@langchain/openai';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { RunnableConfig } from '@langchain/core/runnables';

dotenv.config();

const run = async () => {
	try {
		const PlanExecuteState = Annotation.Root({
			input: Annotation<string>({
				reducer: (x, y) => y ?? x ?? ''
			}),
			plan: Annotation<string[]>({
				reducer: (x, y) => y ?? x ?? []
			}),
			pastSteps: Annotation<[string, string][]>({
				reducer: (x, y) => x.concat(y)
			}),
			response: Annotation<string>({
				reducer: (x, y) => y ?? x
			})
		});

		const tools = [new TavilySearchResults({ maxResults: 3 })];

		const agentExecutor = createReactAgent({
			llm: new ChatOpenAI({ modelName: 'gpt-4o' }),
			tools
		});

		const plan = zodToJsonSchema(
			z.object({
				steps: z.array(z.string()).describe('Different steps to follow, should be in sorted order')
			})
		);

		const planFunction = {
			name: 'plan',
			description: 'This tool is used to plan the steps to follow',
			parameters: plan
		};

		const planTool = {
			type: 'function',
			function: planFunction
		};

		const plannerPrompt = ChatPromptTemplate.fromTemplate(`
            For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            {objective}
        `);

		const model = new ChatOpenAI({ modelName: 'gpt-4o' }).withStructuredOutput(planFunction);

		const planner = plannerPrompt.pipe(model);

		const response = zodToJsonSchema(
			z.object({
				response: z.string().describe('Response to user.')
			})
		);

		const responseTool = {
			type: 'function',
			function: {
				name: 'response',
				description: 'Response to user.',
				parameters: response
			}
		};

		const replannerPrompt = ChatPromptTemplate.fromTemplate(`
            For the given objective, come up with a simple step by step plan. 
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps.
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            Your objective was this:
            {input}

            Your original plan was this:
            {plan}

            You have currently done the following steps:
            {pastSteps}

            Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that and use the 'response' function.
            Otherwise, fill out the plan.  
            Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
        `);

		const parser = new JsonOutputToolsParser();
		const replanner = replannerPrompt
			.pipe(new ChatOpenAI({ modelName: 'gpt-4o' }).bindTools([planTool, responseTool]))
			.pipe(parser);

		const executionStep = async (state: typeof PlanExecuteState.State, config?: RunnableConfig): Promise<Partial<typeof PlanExecuteState.State>> => {
			const task = state.plan[0];
			const input = { messages: [new HumanMessage(task)] };

			const { messages } = await agentExecutor.invoke(input, config);

			return {
				pastSteps: [[task, messages[messages.length - 1].content.toString()]],
				plan: state.plan.slice(1)
			};
		};

		const planStep = async (state: typeof PlanExecuteState.State): Promise<Partial<typeof PlanExecuteState.State>> => {
			const plan = await planner.invoke({ objective: state.input });
			return { plan: plan.steps };
		};

		const replanStep = async (state: typeof PlanExecuteState.State): Promise<Partial<typeof PlanExecuteState.State>> => {
			const output = await replanner.invoke({
				input: state.input,
				plan: state.plan.join('\n'),
				pastSteps: state.pastSteps.map(([step, result]) => `${step}: ${result}`).join('\n')
			});

			const toolCall = output[0];

			if (toolCall.type == 'response') {
				return { response: toolCall.args?.response };
			}

			return { plan: toolCall.args?.steps };
		};

		const shouldEnd = (state: typeof PlanExecuteState.State) => (state.response ? 'true' : 'false');

		const workflow = new StateGraph(PlanExecuteState)
			.addNode('planner', planStep)
			.addNode('agent', executionStep)
			.addNode('replan', replanStep)
			.addEdge(START, 'planner')
			.addEdge('planner', 'agent')
			.addEdge('agent', 'replan')
			.addConditionalEdges('replan', shouldEnd, { true: END, false: 'agent' });

		const app = workflow.compile();

		const config = { recursionLimit: 50 };
		const inputs = {
			input: 'Who is the 2022 NBA Finals MVP and where is his hometown?'
		};

		for await (const event of await app.stream(inputs, config)) {
			console.log(event);
		}
	} catch (error) {
		console.log('%c Line:14 🎂 error', 'color:#2eafb0', error);
	}
};

run();
