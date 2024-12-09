import dotenv from 'dotenv';
import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
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

		const plannerPrompt = ChatPromptTemplate.fromTemplate(`
            For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

            {objective}
        `);

		const model = new ChatOpenAI({ modelName: 'gpt-4o' }).withStructuredOutput(planFunction);

		const planner = plannerPrompt.pipe(model);

		const result = await planner.invoke({
			objective: 'What is the hometown of the current Australia open winner?'
		});

		console.log('%c Line:43 🍎 result', 'color:#3f7cff', result);
	} catch (error) {
		console.log('%c Line:34 🍪 error', 'color:#42b983', error);
	}
};

run();
