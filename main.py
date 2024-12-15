import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI
from taipy.gui import Gui, notify
import taipy.gui.builder as tgb

from PIL import Image
import re
from logger_config import get_logger
from retrieval import load_documents, build_inverted_index, search

logger = get_logger(__name__)
documents = load_documents("data/names.txt")
bm25 = build_inverted_index(documents)


load_dotenv()


def on_init(state):
    state.conv.update_content(state, "")
    state.messages_dict = {}
    state.messages = [
        {
            "role": "assistant",
            "style": "assistant_message",
            "content": "你好! 我是一个基于Qwen的古代服饰提取助手, 请输入需要解析的文本段落: 如头上戴着金丝八宝攒珠髻",
        },
    ]
    state.gpt_messages = []
    new_conv = create_conv(state)
    state.conv.update_content(state, new_conv)


def create_conv(state):
    messages_dict = {}
    with tgb.Page() as conversation:
        for i, message in enumerate(state.messages):
            text = message["content"].replace("<br>", "\n").replace('"', "'")
            messages_dict[f"message_{i}"] = text
            tgb.text(
                "{messages_dict['" + f"message_{i}" + "'] if messages_dict else ''}",
                class_name=f"message_base {message['style']}",
                mode="md",
            )
    state.messages_dict = messages_dict
    return conversation


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def format_queries_as_text(queries_results):
    """
    Format a list of queries and their results into a structured plain text string.

    Args:
        queries_results (List[Tuple[str, List[str]]]): List of (query, results) pairs.

    Returns:
        str: Formatted string in plain text style.
    """
    text_output = ""
    text_output += f"输入文本服饰解析和数据库相关服饰检索结果如下:\n\n"
    for idx, (query, results) in enumerate(queries_results, start=1):
        text_output += f"服饰解析结果 {idx}: {query}\n"
        if results:
            for i, result in enumerate(results, start=1):
                text_output += f" 服饰 {i}: {result}\n"
        else:
            text_output += "未在数据库中找到类似服饰\n"
        text_output += "\n"  # Extra newline for spacing between queries
    return text_output



def query_gpt4o(state):
    if state.query_image_path != "":
        return "暂不支持图片输入~"
    else:
        messages = [
                    {'role': 'user', 
                    'content': f"你需要提取输入文本中的全部衣着或配饰信息, 如从'头上戴着金丝八宝攒珠髻'中提取'金丝八宝攒珠髻', 注意输出时不可以修改原文内容, 应原封不动输出, 如果有多个衣着或配饰信息, 用空格作为分割\n{state.query_message}",
                    }
                ]
        response = state.client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages,
            )
        response_str = response.choices[0].message.content.replace("\n", "")
        resp_list = re.split(r'\s+', response_str)

    state.gpt_messages.append(messages)
    logger.info(f"Query: {state.query_message}")
    logger.info(f"Response: {response.choices[0].message.content}")

    new_resp_list = []  # list(解析文本: str, 检索文本" List[str])

    for resp in resp_list:
        search_resp = search(resp, bm25, documents, 3)
        # search_resp是一个tuple，第一个元素是文档内容，第二个元素是得分
        # TODO: 添加检索结果对应的图片地址
        retrieved_docs = [doc[0] for doc in search_resp]
        new_resp_list.append([resp, retrieved_docs])
    # 根据检索结果整理回复格式并生成回复消息
    resp_text = format_queries_as_text(new_resp_list)

    return resp_text


def send_message(state):
    if state.query_image_path == "":
        state.messages.append(
            {
                "role": "user",
                "style": "user_message",
                "content": state.query_message,
            }
        )
    else:
        state.messages.append(
            {
                "role": "user",
                "style": "user_message",
                "content": f"{state.query_message}\n![user_image]({state.query_image_path})",
            }
        )
    state.conv.update_content(state, create_conv(state))
    notify(state, "info", "Sending message...")
    state.messages.append(
        {
            "role": "assistant",
            "style": "assistant_message",
            "content": f"{query_gpt4o(state)}",
        }
    )
    # state.messages.append(
    #     {
    #         "role": "assistant",
    #         "style": "assistant_message",
    #         "content": f"{query_gpt4o(state)}![assistant_image](images/example_0.png)",
    #     }
    # )
    state.conv.update_content(state, create_conv(state))
    state.query_message = ""
    state.query_image_path = ""


def upload_image(state):
    try:
        global index
        image = Image.open(state.query_image_path)
        image.thumbnail((300, 300))
        image.save(f"images/example_{index}.png")
        state.query_image_path = f"images/example_{index}.png"
        index = index + 1
    except:
        notify(
            state,
            "error",
            f"Please make sure your image is under 1MB",
        )


def reset_chat(state):
    state.messages = []
    state.gpt_messages = []
    state.query_message = ""
    state.query_image_path = ""
    state.conv.update_content(state, create_conv(state))
    on_init(state)


if __name__ == "__main__":
    index = 0
    query_image_path = ""
    query_message = ""
    messages = []
    gpt_messages = []
    messages_dict = {}
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    with tgb.Page() as page:
        with tgb.layout(columns="300px 1"):
            with tgb.part(class_name="sidebar"):
                tgb.text("## Taipy x GPT-4o", mode="md")
                tgb.button(
                    "New Conversation",
                    class_name="fullwidth plain",
                    id="reset_app_button",
                    on_action=reset_chat,
                )
                tgb.html("br")
                tgb.image(
                    content="{query_image_path}", width="250px", class_name="image_preview"
                )

            with tgb.part(class_name="p1"):
                tgb.part(partial="{conv}", height="600px", class_name="card card_chat")
                with tgb.part("card mt1"):
                    tgb.input(
                        "{query_message}",
                        on_action=send_message,
                        change_delay=-1,
                        label="Write your message:",
                        class_name="fullwidth",
                    )
                    tgb.file_selector(
                        content="{query_image_path}",
                        on_action=upload_image,
                        extensions=".jpg,.jpeg,.png",
                        label="Upload an image",
                    )
                    tgb.text("Max file size: 1MB")
    gui = Gui(page)
    conv = gui.add_partial("")
    gui.run(title="🤖Taipy x GPT-4o", dark_mode=False, margin="0px", debug=True, use_reloader=True)
