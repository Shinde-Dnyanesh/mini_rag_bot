import os
from retriever import Retriever
from llm_query import generate
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

def start_bot(token: str):
    retriever = Retriever()

    async def start_handler(update: Update, context:ContextTypes.DEFAULT_TYPE):
        welcome_text = (
            "Hi! I’m *Avivo Assistant*, your smart RAG-powered helper.\n\n"
            "You can ask me questions about company info or policies.\n"
            "Try typing:\n"
            "`/ask What is the refund policy?`")
        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = (
            " *Available Commands:*\n\n"
            "/start - Welcome message\n"
            "/help - Show this help menu\n"
            "/ask <question> - Ask me a question based on company documents\n\n"
            "You can also just type your question directly — no need to use /ask every time!")
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = " ".join(context.args)
        if not q:
            await update.message.reply_text("Usage: /ask <your question>")
            return

        contexts = retriever.retrieve(q)
        answer = generate(q, contexts)
        sources = ", ".join({c["source"] for c in contexts})
        await update.message.reply_text(f"{answer}\n\nSources: {sources}")

    async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.message.text.strip()
        if q.startswith("/"):
            return
        await update.message.chat.send_action(action=ChatAction.TYPING)
        contexts = retriever.retrieve(q)
        answer = generate(q, contexts)
        sources = ", ".join({c["source"] for c in contexts})
        await update.message.reply_text(f"{answer}\n\nSources: {sources}")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help", help_handler))
    app.add_handler(CommandHandler("ask", ask_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    print("Bot running... open Telegram and send /start or ask a question.")
    app.run_polling()

if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Please set TELEGRAM_BOT_TOKEN environment variable and run again.")
    else:
        start_bot(token)
