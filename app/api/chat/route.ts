import { streamText, UIMessage, convertToModelMessages } from 'ai';
import { openai } from '@ai-sdk/openai';
import { anthropic } from '@ai-sdk/anthropic';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const {
    messages,
    model,
    webSearch,
  }: { 
    messages: UIMessage[]; 
    model: string; 
    webSearch: boolean;
  } = await req.json();

  // Validate that we have API keys configured
  if (!process.env.OPENAI_API_KEY && !process.env.ANTHROPIC_API_KEY) {
    return new Response(
      JSON.stringify({ 
        error: 'No AI provider API keys configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment variables.' 
      }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }

  try {
    // Determine which model to use
    let modelToUse = model;
    
    // If web search is enabled, use a search-enabled model if available
    if (webSearch && process.env.OPENAI_API_KEY) {
      // Use GPT-4o for web-enabled searches
      modelToUse = 'openai/gpt-4o';
    }

    // Parse the model string to get provider and model name
    const [provider, modelName] = modelToUse.split('/');
    
    // Select the appropriate model instance based on provider
    let modelInstance;
    if (provider === 'openai') {
      modelInstance = openai(modelName || 'gpt-4o');
    } else if (provider === 'anthropic') {
      modelInstance = anthropic(modelName || 'claude-3-5-sonnet-20241022');
    } else if (provider === 'deepseek') {
      modelInstance = openai(modelName || 'deepseek-r1');
    } else {
      // Default to openai gpt-4o if provider not recognized
      modelInstance = openai('gpt-4o');
    }

    const result = streamText({
      model: modelInstance,
      messages: convertToModelMessages(messages),
      system:
        'You are a helpful assistant that can answer questions and help with tasks',
    });
    
    // send sources and reasoning back to the client
    return result.toUIMessageStreamResponse({
      sendSources: true,
      sendReasoning: true,
    });
  } catch (error) {
    console.error('Error in chat API:', error);
    return new Response(
      JSON.stringify({ 
        error: 'An error occurred while processing your request.' 
      }),
      { 
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}