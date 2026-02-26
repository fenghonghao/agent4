"""
MemoryManager单元测试
覆盖Message类和MemoryManager的所有核心功能
"""
import pytest
import os
import json
import time
from unittest.mock import MagicMock, patch, mock_open

# 必须在导入core模块前mock litellm
import sys
from unittest.mock import MagicMock
mock_litellm = MagicMock()
mock_litellm.token_counter.return_value = 100
mock_litellm.suppress_instrumentation = True
sys.modules['litellm'] = mock_litellm

from core.agents.agent_memory.memory import Message, MemoryManager


@pytest.mark.unit
class TestMessage:
    """测试Message类的基本功能"""

    def test_message_creation(self):
        """测试Message对象创建"""
        msg = Message(role="user", content="Hello", pinned=True)
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.pinned is True
        assert msg.image_base64 is None
        assert msg.timestamp is not None

    def test_message_creation_with_image(self):
        """测试带图片的Message创建"""
        img_data = "base64data"
        msg = Message(role="user", content="Screenshot", image_base64=img_data)
        assert msg.image_base64 == img_data
        assert msg.role == "user"

    def test_message_creation_with_tool_calls(self):
        """测试带tool_calls的Message创建"""
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        msg = Message(role="assistant", tool_calls=tool_calls)
        assert msg.tool_calls == tool_calls
        assert msg.role == "assistant"

    def test_estimate_tokens_text(self):
        """测试纯文本token估算"""
        msg = Message(role="user", content="Hello world")
        tokens = msg.estimate_tokens()
        # 默认mock返回100
        assert tokens == 100

    def test_estimate_tokens_with_image(self):
        """测试图片token估算(固定1100)"""
        msg = Message(role="user", content="Screenshot", image_base64="imgdata")
        tokens = msg.estimate_tokens()
        # 图片固定1100 + 文本100
        assert tokens == 1200

    def test_estimate_tokens_with_tool_calls(self):
        """测试tool_calls token估算"""
        tool_calls = [{"id": "call_1"}, {"id": "call_2"}]
        msg = Message(role="assistant", tool_calls=tool_calls)
        tokens = msg.estimate_tokens()
        # 2个tool calls * 75 = 150
        assert tokens == 150

    def test_estimate_tokens_with_function_call(self):
        """测试旧版function_call token估算"""
        func_call = {"name": "test", "arguments": "{}"}
        msg = Message(role="assistant", function_call=func_call)
        tokens = msg.estimate_tokens()
        # 1个function call = 75
        assert tokens == 75

    def test_to_dict_text(self):
        """测试普通文本消息转dict"""
        msg = Message(role="user", content="Hello")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello"}

    def test_to_dict_with_image(self):
        """测试图片消息转dict"""
        img_data = "base64data"
        msg = Message(role="user", content="Screenshot", image_base64=img_data)
        result = msg.to_dict()
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert result["content"][0] == {"type": "text", "text": "Screenshot"}
        assert result["content"][1]["type"] == "image_url"
        assert "base64data" in result["content"][1]["image_url"]["url"]

    def test_to_dict_image_only(self):
        """测试只有图片的消息转dict"""
        img_data = "base64data"
        msg = Message(role="user", image_base64=img_data)
        result = msg.to_dict()
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "image_url"

    def test_to_dict_tool_role(self):
        """测试tool role消息转dict"""
        msg = Message(role="tool", content="Result", tool_call_id="call_123")
        result = msg.to_dict()
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "Result"

    def test_to_dict_with_tool_calls(self):
        """测试assistant带tool_calls转dict"""
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        msg = Message(role="assistant", tool_calls=tool_calls, content="Thinking")
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert result["tool_calls"] == tool_calls
        assert result["content"] == "Thinking"

    def test_to_dict_with_tool_calls_no_content(self):
        """测试assistant带tool_calls但无content"""
        tool_calls = [{"id": "call_1"}]
        msg = Message(role="assistant", tool_calls=tool_calls)
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert result["tool_calls"] == tool_calls
        assert "content" not in result


@pytest.mark.unit
class TestMemoryManagerInit:
    """测试MemoryManager初始化"""

    def test_init_default_values(self, temp_dir):
        """测试默认参数初始化"""
        with patch.dict(os.environ, {"CodeAgent_MODEL": "gpt-4o"}):
            mm = MemoryManager(agent_name="test_agent", save_dir=temp_dir)
            assert mm.agent_name == "test_agent"
            assert mm.max_tokens == 8000
            assert mm.keep_last_screenshots == 2
            assert mm.keep_function_calls == 5
            assert mm.history == []
            assert mm.insights == {}
            assert mm.function_stats == {}

    def test_init_custom_values(self, temp_dir):
        """测试自定义参数初始化"""
        mm = MemoryManager(
            agent_name="custom_agent",
            max_tokens=5000,
            keep_last_screenshots=3,
            keep_function_calls=10,
            save_dir=temp_dir,
            model="custom-model"
        )
        assert mm.agent_name == "custom_agent"
        assert mm.max_tokens == 5000
        assert mm.keep_last_screenshots == 3
        assert mm.keep_function_calls == 10
        assert mm.model == "custom-model"

    def test_init_from_env(self, temp_dir):
        """测试从环境变量读取model"""
        with patch.dict(os.environ, {"CodeAgent_MODEL": "env-model"}):
            mm = MemoryManager(agent_name="test", save_dir=temp_dir)
            assert mm.model == "env-model"

    def test_init_default_model(self, temp_dir):
        """测试默认model值"""
        with patch.dict(os.environ, {}, clear=True):
            mm = MemoryManager(agent_name="test", save_dir=temp_dir)
            assert mm.model == "gpt-4o"


@pytest.mark.unit
class TestVisualPruning:
    """测试视觉遗忘（图片修剪）功能"""

    def test_no_pruning_when_under_limit(self, temp_dir):
        """图片数未超限制不修剪"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_last_screenshots=5
        )
        # 添加3张图片（少于5张限制）
        for i in range(3):
            mm.add("user", f"Image {i}", image_base64=f"img{i}")

        # 所有图片应该保留
        img_msgs = [m for m in mm.history if m.image_base64]
        assert len(img_msgs) == 3

    def test_pruning_removes_oldest_images(self, temp_dir):
        """按FIFO修剪最早图片"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_last_screenshots=2
        )
        # 添加5张图片
        for i in range(5):
            mm.add("user", f"Image {i}", image_base64=f"img{i}")

        # 只保留最后2张
        img_msgs = [m for m in mm.history if m.image_base64]
        assert len(img_msgs) == 2
        # 保留的是最后两张
        assert img_msgs[0].image_base64 == "img3"
        assert img_msgs[1].image_base64 == "img4"

    def test_pruning_preserves_message_structure(self, temp_dir):
        """保留消息结构，仅清空图片"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_last_screenshots=2
        )
        # 添加5张图片
        for i in range(5):
            mm.add("user", f"Image {i}", image_base64=f"img{i}")

        # 检查被修剪的消息
        removed_msgs = [m for m in mm.history if not m.image_base64 and "[截图已移除]" in (m.content or "")]
        assert len(removed_msgs) == 3
        # 消息结构保留
        assert removed_msgs[0].content == "[截图已移除] Image 0"

    def test_pruning_with_keep_zero(self, temp_dir):
        """keep_last_screenshots=0时跳过视觉修剪"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_last_screenshots=0
        )
        # 添加5张图片
        for i in range(5):
            mm.add("user", f"Image {i}", image_base64=f"img{i}")

        # keep=0时不应该修剪
        # 注意：代码逻辑是if self.keep_last_screenshots > 0才执行
        img_msgs = [m for m in mm.history if m.image_base64]
        # 由于代码中keep_last_screenshots > 0才修剪，所以这里应该是5张
        # 但实际代码逻辑中，keep=0表示不限制，所以全部保留
        assert len(img_msgs) == 5


@pytest.mark.unit
class TestFunctionPruning:
    """测试Function Call修剪功能"""

    def test_function_pairs_identification(self, temp_dir):
        """正确识别function call配对"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_function_calls=5
        )
        # 添加一个function call配对
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        mm.add_function_call(tool_calls)
        mm.add_function_result("call_1", "test", "result")

        # 应该识别出1个配对
        func_call_pairs = []
        i = 0
        while i < len(mm.history):
            msg = mm.history[i]
            if msg.tool_calls:
                pair_indices = [i]
                for j in range(i + 1, len(mm.history)):
                    if mm.history[j].role == "tool":
                        pair_indices.append(j)
                    else:
                        break
                func_call_pairs.append(pair_indices)
            i += 1

        assert len(func_call_pairs) == 1
        assert len(func_call_pairs[0]) == 2  # assistant + tool

    def test_old_pairs_unpinned(self, temp_dir):
        """旧配对被标记为可删除（pinned=False）"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_function_calls=1  # 只保留1组
        )
        # 添加2个function call配对
        for i in range(2):
            tool_calls = [{"id": f"call_{i}", "function": {"name": f"tool_{i}"}}]
            mm.add_function_call(tool_calls)
            mm.add_function_result(f"call_{i}", f"tool_{i}", f"result_{i}")

        # 检查所有消息都不应该是pinned
        for msg in mm.history:
            assert msg.pinned is False

    def test_multiple_tools_per_call(self, temp_dir):
        """一个assistant调用多个tool"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            keep_function_calls=5
        )
        # 一个assistant消息调用2个tools
        tool_calls = [
            {"id": "call_1", "function": {"name": "tool_a"}},
            {"id": "call_2", "function": {"name": "tool_b"}}
        ]
        mm.add_function_call(tool_calls)
        mm.add_function_result("call_1", "tool_a", "result_a")
        mm.add_function_result("call_2", "tool_b", "result_b")

        # 检查配对识别
        assert len(mm.history) == 3  # assistant + 2 tools
        assert mm.history[0].role == "assistant"
        assert mm.history[1].role == "tool"
        assert mm.history[2].role == "tool"


@pytest.mark.unit
class TestTokenPruning:
    """测试Token滑动窗口修剪"""

    def test_no_pruning_under_limit(self, temp_dir):
        """token未超限不修剪"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=1000
        )
        # 添加2条消息（每条约100 tokens）
        mm.add("user", "Hello")
        mm.add("assistant", "Hi there")

        assert len(mm.history) == 2

    def test_pruning_removes_oldest_unpinned(self, temp_dir):
        """删除最早非pinned消息"""
        # 每条消息100 tokens，限制300 tokens
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=250  # 最多保留2条消息
        )
        # 添加3条消息
        mm.add("user", "Message 1")
        mm.add("user", "Message 2")
        mm.add("user", "Message 3")

        # 最旧的消息应该被删除
        assert len(mm.history) == 2
        assert mm.history[0].content == "Message 2"
        assert mm.history[1].content == "Message 3"

    def test_pruning_respects_pinned(self, temp_dir):
        """pinned消息不被删除"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=250
        )
        # 添加3条消息，第一条pinned
        mm.add("user", "Pinned message", pinned=True)
        mm.add("user", "Message 2")
        mm.add("user", "Message 3")

        # pinned消息应该保留，删除Message 2
        assert len(mm.history) == 2
        assert mm.history[0].content == "Pinned message"
        assert mm.history[0].pinned is True
        assert mm.history[1].content == "Message 3"

    def test_pruning_extreme_case(self, temp_dir):
        """全pinned时强制删除最旧一条"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=200  # 只能保留1条
        )
        # 添加3条pinned消息
        mm.add("user", "Message 1", pinned=True)
        mm.add("user", "Message 2", pinned=True)
        mm.add("user", "Message 3", pinned=True)

        # 极端情况下也会删除（通过pop(0)）
        assert len(mm.history) == 2
        assert mm.history[0].content == "Message 2"
        assert mm.history[1].content == "Message 3"

    def test_keep_at_least_one_message(self, temp_dir):
        """至少保留一条消息"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=50  # 只能保留0条
        )
        # 添加1条消息
        mm.add("user", "Only message")

        # 至少保留一条
        assert len(mm.history) == 1


@pytest.mark.unit
class TestAddMethods:
    """测试各种add方法"""

    def test_add_message(self, temp_dir):
        """添加普通消息"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add("user", "Hello", pinned=True)

        assert len(mm.history) == 1
        assert mm.history[0].role == "user"
        assert mm.history[0].content == "Hello"
        assert mm.history[0].pinned is True

    def test_add_function_call_updates_stats(self, temp_dir):
        """添加function call更新统计"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        tool_calls = [
            {"id": "call_1", "function": {"name": "tool_a"}},
            {"id": "call_2", "function": {"name": "tool_b"}}
        ]
        mm.add_function_call(tool_calls, "Thinking")

        assert len(mm.history) == 1
        assert mm.history[0].role == "assistant"
        assert mm.history[0].tool_calls == tool_calls
        # 统计更新
        assert mm.function_stats["tool_a"] == 1
        assert mm.function_stats["tool_b"] == 1

    def test_add_function_result(self, temp_dir):
        """添加function执行结果"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add_function_result("call_123", "test_tool", "result data")

        assert len(mm.history) == 1
        assert mm.history[0].role == "tool"
        assert mm.history[0].tool_call_id == "call_123"
        assert mm.history[0].content == "result data"

    def test_add_triggers_pruning(self, temp_dir):
        """添加消息自动触发修剪"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=10000,  # 足够高以避免token修剪(图片消息约1200 tokens)
            keep_last_screenshots=1
        )
        # 添加2张图片，应该触发视觉修剪
        mm.add("user", "Image 1", image_base64="img1")
        mm.add("user", "Image 2", image_base64="img2")

        # 检查被修剪的图片（第一张被移除，第二张保留）
        # 遍历history找到被修剪的消息
        pruned_msgs = [m for m in mm.history if m.image_base64 is None and "[截图已移除]" in (m.content or "")]
        assert len(pruned_msgs) == 1
        assert "[截图已移除] Image 1" in pruned_msgs[0].content


@pytest.mark.unit
class TestContextBuilding:
    """测试Context构建"""

    def test_get_context_with_system_prompt(self, temp_dir):
        """包含system prompt"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.set_system_prompt("You are a helpful assistant")
        mm.add("user", "Hello")

        context = mm.get_context()

        assert len(context) == 2
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "You are a helpful assistant"
        assert context[1]["role"] == "user"

    def test_get_context_with_insights(self, temp_dir):
        """注入insights到system prompt"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.set_system_prompt("You are a helpful assistant")
        mm.add_insight("key1", "value1")
        mm.add_insight("key2", "value2")

        context = mm.get_context()

        sys_content = context[0]["content"]
        assert "[长期记忆/Insights]" in sys_content
        assert "key1: value1" in sys_content
        assert "key2: value2" in sys_content

    def test_get_context_with_function_stats(self, temp_dir):
        """注入function统计"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.set_system_prompt("You are a helpful assistant")
        mm.function_stats = {"tool_a": 10, "tool_b": 5}

        context = mm.get_context()

        sys_content = context[0]["content"]
        assert "[常用工具统计]" in sys_content
        assert "tool_a: 10次" in sys_content

    def test_get_context_empty_history(self, temp_dir):
        """空history处理"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.set_system_prompt("You are a helpful assistant")

        context = mm.get_context()

        assert len(context) == 1
        assert context[0]["role"] == "system"

    def test_get_context_no_system_prompt(self, temp_dir):
        """无system prompt时"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add("user", "Hello")

        context = mm.get_context()

        assert len(context) == 1
        assert context[0]["role"] == "user"


@pytest.mark.unit
class TestInsights:
    """测试Insights功能"""

    def test_add_insight(self, temp_dir):
        """添加insight"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add_insight("topic1", "knowledge1")

        assert mm.insights["topic1"] == "knowledge1"

    def test_add_insight_overwrite(self, temp_dir):
        """同名insight覆盖"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add_insight("topic1", "knowledge1")
        mm.add_insight("topic1", "knowledge2")

        assert mm.insights["topic1"] == "knowledge2"


@pytest.mark.unit
class TestClear:
    """测试清理功能"""

    def test_clear_short_term(self, temp_dir):
        """清空短期记忆保留insights"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add("user", "Hello")
        mm.add_insight("topic", "knowledge")
        mm.function_stats = {"tool": 5}

        mm.clear_short_term()

        # 短期记忆清空
        assert mm.history == []
        # 长期记忆保留
        assert mm.insights == {"topic": "knowledge"}


@pytest.mark.unit
class TestPersistence:
    """测试持久化功能"""

    def test_save_and_load_insights(self, temp_dir):
        """测试insights的保存和加载"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add_insight("topic1", "value1")
        mm.add_insight("topic2", "value2")

        # 创建新实例，应该加载之前的insights
        mm2 = MemoryManager(agent_name="test", save_dir=temp_dir)
        assert mm2.insights["topic1"] == "value1"
        assert mm2.insights["topic2"] == "value2"

    def test_save_and_load_function_stats(self, temp_dir):
        """测试function_stats的保存和加载"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        tool_calls = [{"id": "call_1", "function": {"name": "tool_a"}}]
        mm.add_function_call(tool_calls)

        # 创建新实例，应该加载之前的stats
        mm2 = MemoryManager(agent_name="test", save_dir=temp_dir)
        assert mm2.function_stats["tool_a"] == 1

    def test_get_function_stats(self, temp_dir):
        """测试获取function统计"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.function_stats = {"tool_a": 5, "tool_b": 3}

        stats = mm.get_function_stats()
        assert stats == {"tool_a": 5, "tool_b": 3}

    def test_get_function_stats_returns_copy(self, temp_dir):
        """测试get_function_stats返回副本"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.function_stats = {"tool": 1}

        stats = mm.get_function_stats()
        stats["tool"] = 100  # 修改返回的副本

        # 原始数据不应该改变
        assert mm.function_stats["tool"] == 1


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""

    def test_add_none_content(self, temp_dir):
        """添加None content"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add("user", None)

        msg_dict = mm.history[0].to_dict()
        assert msg_dict["content"] == ""

    def test_message_to_dict_empty_content(self, temp_dir):
        """Message to_dict空content处理"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add("user", "")

        msg_dict = mm.history[0].to_dict()
        assert msg_dict["content"] == ""

    def test_estimate_tokens_empty_message(self, temp_dir):
        """估算空消息token"""
        msg = Message(role="user", content=None)
        tokens = msg.estimate_tokens()
        assert tokens == 0

    def test_pruning_with_mixed_messages(self, temp_dir):
        """混合消息类型的修剪"""
        mm = MemoryManager(
            agent_name="test",
            save_dir=temp_dir,
            max_tokens=300,
            keep_last_screenshots=1
        )
        # 添加文本和图片混合
        mm.add("user", "Text 1")
        mm.add("user", "Image 1", image_base64="img1")
        mm.add("user", "Text 2")
        mm.add("user", "Image 2", image_base64="img2")

        # 检查图片修剪和文本保留
        img_count = len([m for m in mm.history if m.image_base64])
        assert img_count == 1  # 只保留1张图片

    def test_function_call_with_empty_tool_calls(self, temp_dir):
        """空tool_calls列表"""
        mm = MemoryManager(agent_name="test", save_dir=temp_dir)
        mm.add_function_call([])

        assert len(mm.history) == 1
        assert mm.history[0].tool_calls == []
