import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { JsonOutputToolsParser } from '@langchain/core/output_parsers/openai_tools';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

dotenv.config();

const run = async () => {
	try {
		const plan = zodToJsonSchema(
			z.object({ steps: z.array(z.string()).describe('Different steps to follow, should be in sorted order') })
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

		const plannerResult = await planner.invoke({
			objective: 'What is the hometown of the current Australia open winner?'
		});
		console.log('%c Line:35 🍕 plannerResult', 'color:#6ec1c2', plannerResult);

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

            You have currently done the follow steps:
            {pastSteps}

            Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that and use the 'response' function.
            Otherwise, fill out the plan.  
            Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan.
        `);

		const parser = new JsonOutputToolsParser();
		const replanner = replannerPrompt
			.pipe(new ChatOpenAI({ model: 'gpt-4o' }).bindTools([planTool, responseTool]))
			.pipe(parser);

		const replannerResult = await replanner.invoke({
			input: 'What is the hometown of the current Australia open winner?',
			plan: plannerResult.steps.join(','),
			pastSteps: []
		});

		console.log('%c Line:79 🥚 replannerResult', 'color:#ffdd4d', replannerResult);
	} catch (error) {
		console.log('%c Line:34 🍪 error', 'color:#42b983', error);
	}
};

run();
