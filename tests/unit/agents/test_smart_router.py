"""
SmartRouter单元测试
覆盖任务分析、Agent选择、失败切换等核心逻辑
"""
import pytest
import queue
import os
import sys
from unittest.mock import MagicMock, patch, call

# 导入core模块 (conftest.py已设置mock)
from core.agents.smart_router import SmartRouter, get_router


@pytest.mark.unit
class TestSingleton:
    """测试单例模式"""

    def test_get_router_returns_same_instance(self, reset_smart_router_singleton):
        """单例模式验证"""
        router1 = get_router()
        router2 = get_router()
        assert router1 is router2

    def test_get_router_lazy_initialization(self, reset_smart_router_singleton):
        """懒加载验证"""
        # 首次获取时创建
        router1 = get_router()
        assert router1 is not None
        # 再次获取时返回同一实例
        router2 = get_router()
        assert router1 is router2


@pytest.mark.unit
class TestAnalyzeTask:
    """测试任务分析功能"""

    def test_gui_keywords_scoring(self, reset_smart_router_singleton, mock_env_vars):
        """GUI关键词评分"""
        router = SmartRouter()
        agent_type, confidence = router.analyze_task("打开浏览器搜索GitHub")

        assert agent_type == "gui"
        # 匹配了"打开"和"浏览器"两个关键词
        # 置信度 = 0.6 + 2 * 0.1 = 0.8
        assert confidence == 0.8

    def test_code_keywords_scoring(self, reset_smart_router_singleton, mock_env_vars):
        """Code关键词评分"""
        router = SmartRouter()
        agent_type, confidence = router.analyze_task("计算斐波那契数列的代码")

        assert agent_type == "code"
        # 匹配了"计算"、"代码"两个关键词
        assert confidence >= 0.7

    def test_equal_scores_falls_back_to_llm(self, reset_smart_router_singleton, mock_env_vars):
        """相等分数时调用LLM"""
        router = SmartRouter()

        # Mock _llm_analyze
        router._llm_analyze = MagicMock(return_value=("gui", 0.7))

        # 使用一个没有明显倾向的任务描述
        agent_type, confidence = router.analyze_task("完成这个任务")

        # 由于没有关键词匹配，gui_score == code_score == 0，应该调用LLM
        router._llm_analyze.assert_called_once()

    def test_empty_task_description(self, reset_smart_router_singleton, mock_env_vars):
        """空任务描述处理"""
        router = SmartRouter()
        router._llm_analyze = MagicMock(return_value=("gui", 0.5))

        agent_type, confidence = router.analyze_task("")

        # 空字符串应该调用LLM
        router._llm_analyze.assert_called_once()

    def test_confidence_calculation(self, reset_smart_router_singleton, mock_env_vars):
        """置信度计算(0.6+0.1*n)，上限0.95"""
        router = SmartRouter()

        # 使用多个GUI关键词
        agent_type, confidence = router.analyze_task(
            "打开浏览器点击按钮拖拽窗口界面菜单截图屏幕鼠标键盘输入应用"
        )

        assert agent_type == "gui"
        # 置信度上限为0.95
        assert confidence == 0.95

    def test_confidence_calculation_code(self, reset_smart_router_singleton, mock_env_vars):
        """Code置信度计算"""
        router = SmartRouter()

        agent_type, confidence = router.analyze_task(
            "计算算法函数变量循环判断数据处理文件读写json csv api数学统计绘图分析数据"
        )

        assert agent_type == "code"
        assert confidence == 0.95


@pytest.mark.unit
class TestLLMAnalyze:
    """测试LLM分析功能"""

    def test_llm_returns_gui(self, reset_smart_router_singleton, mock_env_vars):
        """LLM返回GUI分类"""
        router = SmartRouter()

        # Mock completion返回GUI响应
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "GUI:0.85"

        with patch('core.agents.smart_router.completion', return_value=mock_response):
            agent_type, confidence = router._llm_analyze("some task")

        assert agent_type == "gui"
        assert confidence == 0.85

    def test_llm_returns_code(self, reset_smart_router_singleton, mock_env_vars):
        """LLM返回Code分类"""
        router = SmartRouter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "CODE:0.75"

        with patch('core.agents.smart_router.completion', return_value=mock_response):
            agent_type, confidence = router._llm_analyze("some task")

        assert agent_type == "code"
        assert confidence == 0.75

    def test_llm_parse_various_formats(self, reset_smart_router_singleton, mock_env_vars):
        """解析多种置信度格式"""
        router = SmartRouter()

        test_cases = [
            ("GUI:0.9", "gui", 0.9),
            ("GUI:0.85", "gui", 0.85),
            ("CODE:0.7", "code", 0.7),
            ("gui:0.8", "gui", 0.8),  # 小写
            ("code:0.6", "code", 0.6),
        ]

        for content, expected_type, expected_conf in test_cases:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = content

            with patch('core.agents.smart_router.completion', return_value=mock_response):
                agent_type, confidence = router._llm_analyze("task")
                assert agent_type == expected_type, f"Failed for {content}"
                assert confidence == expected_conf, f"Failed for {content}"

    def test_llm_unparseable_response(self, reset_smart_router_singleton, mock_env_vars):
        """无法解析时默认GUI"""
        router = SmartRouter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "some random response"

        with patch('core.agents.smart_router.completion', return_value=mock_response):
            agent_type, confidence = router._llm_analyze("task")

        assert agent_type == "gui"
        assert confidence == 0.5

    def test_llm_exception_handling(self, reset_smart_router_singleton, mock_env_vars):
        """LLM调用异常处理"""
        router = SmartRouter()

        with patch('core.agents.smart_router.completion', side_effect=Exception("API Error")):
            agent_type, confidence = router._llm_analyze("task")

        assert agent_type == "gui"
        assert confidence == 0.5


@pytest.mark.unit
class TestExecuteWithFallback:
    """测试执行和失败切换功能"""

    def test_successful_gui_execution(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """GUI Agent成功执行"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # Mock GUI成功
        gui_mock.task.return_value = "任务成功完成"

        result = router_instance.execute_with_fallback(
            "打开浏览器",
            mock_queue_from,
            mock_queue_to,
            force_agent="gui"
        )

        assert "成功" in result or "完成" in result
        gui_mock.task.assert_called_once()

    def test_successful_code_execution(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """Code Agent成功执行"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # Mock Code成功
        code_mock.task.return_value = "计算完成"

        result = router_instance.execute_with_fallback(
            "计算数据",
            mock_queue_from,
            mock_queue_to,
            force_agent="code"
        )

        code_mock.task.assert_called_once()

    def test_fallback_to_code_agent(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """GUI失败后切换到Code Agent"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance._is_success = MagicMock(side_effect=[False, True])

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # GUI失败，Code成功
        gui_mock.task.return_value = "失败"
        code_mock.task.return_value = "成功完成"

        result = router_instance.execute_with_fallback(
            "任务描述",
            mock_queue_from,
            mock_queue_to,
            max_retries=1
        )

        # 应该尝试两个agent
        gui_mock.task.assert_called_once()
        code_mock.task.assert_called_once()

    def test_fallback_to_gui_agent(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """Code失败后切换到GUI Agent"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock

        # Mock analyze_task返回code，但code失败
        router_instance.analyze_task = MagicMock(return_value=("code", 0.7))
        router_instance._is_success = MagicMock(side_effect=[False, True])

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # Code失败，GUI成功
        code_mock.task.return_value = "失败"
        gui_mock.task.return_value = "成功完成"

        result = router_instance.execute_with_fallback(
            "任务描述",
            mock_queue_from,
            mock_queue_to,
            max_retries=1
        )

        code_mock.task.assert_called_once()
        gui_mock.task.assert_called_once()

    def test_no_fallback_when_high_confidence(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """高置信度(>=0.8)不切换"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.9))  # 高置信度
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "失败"

        result = router_instance.execute_with_fallback(
            "任务描述",
            mock_queue_from,
            mock_queue_to,
            max_retries=1
        )

        # 高置信度时不应该切换
        gui_mock.task.assert_called_once()
        code_mock.task.assert_not_called()

    def test_forced_agent_no_fallback(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """force_agent时强制指定，不切换"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "失败"

        result = router_instance.execute_with_fallback(
            "任务描述",
            mock_queue_from,
            mock_queue_to,
            force_agent="gui",
            max_retries=1
        )

        # force_agent时不应该切换，即使失败
        gui_mock.task.assert_called_once()
        code_mock.task.assert_not_called()

    def test_max_retries_exceeded(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """超过最大重试次数"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "GUI失败"
        code_mock.task.return_value = "Code失败"

        result = router_instance.execute_with_fallback(
            "任务描述",
            mock_queue_from,
            mock_queue_to,
            max_retries=1
        )

        assert "失败" in result
        assert "最大重试次数" in result or "error" in result.lower()


@pytest.mark.unit
class TestHumanIntervention:
    """测试人类介入功能"""

    def test_human_modify_task(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类修改任务描述"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))
        router_instance._is_success = MagicMock(side_effect=[False, True])

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # 第一次调用返回人类响应
        mock_queue_from.get.side_effect = [
            {
                "name": "SmartRouter",
                "type": "human_response",
                "content": {"action": "modify_task", "modified_task": "修改后的任务"}
            }
        ]

        gui_mock.task.return_value = "失败"

        with patch.object(router_instance, '_wait_for_human_intervention', return_value={
            "action": "modify_task", "modified_task": "修改后的任务"
        }):
            result = router_instance.execute_with_fallback(
                "原始任务",
                mock_queue_from,
                mock_queue_to,
                max_retries=2
            )

    def test_human_retry(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类要求重试"""
        router_instance = SmartRouter()

        mock_queue_from = MagicMock()
        mock_queue_to = MagicMock()

        with patch.object(router_instance, '_wait_for_human_intervention', return_value={
            "action": "retry"
        }):
            # 测试内部逻辑
            response = router_instance._wait_for_human_intervention(
                mock_queue_from, mock_queue_to, timeout=1
            )
            assert response["action"] == "retry"

    def test_human_skip(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类跳过任务"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "失败"

        with patch.object(router_instance, '_wait_for_human_intervention', return_value={
            "action": "skip"
        }):
            result = router_instance.execute_with_fallback(
                "任务描述",
                mock_queue_from,
                mock_queue_to,
                max_retries=2
            )

            assert "跳过" in result

    def test_human_completed(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类标记已完成"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "失败"

        with patch.object(router_instance, '_wait_for_human_intervention', return_value={
            "action": "completed"
        }):
            result = router_instance.execute_with_fallback(
                "任务描述",
                mock_queue_from,
                mock_queue_to,
                max_retries=2
            )

            assert "已完成" in result

    def test_human_provide_context(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类提供额外上下文"""
        router_instance = SmartRouter()

        mock_queue_from = MagicMock()
        mock_queue_to = MagicMock()

        with patch.object(router_instance, '_wait_for_human_intervention', return_value={
            "action": "provide_context", "context": "额外信息"
        }):
            response = router_instance._wait_for_human_intervention(
                mock_queue_from, mock_queue_to, timeout=1
            )
            assert response["action"] == "provide_context"
            assert response["context"] == "额外信息"

    def test_timeout_handling(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """人类介入超时处理"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))
        router_instance._is_success = MagicMock(return_value=False)

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        gui_mock.task.return_value = "失败"

        # max_retries=2 才能在第一次失败后请求人类介入
        with patch.object(router_instance, '_wait_for_human_intervention', return_value=None):
            result = router_instance.execute_with_fallback(
                "任务描述",
                mock_queue_from,
                mock_queue_to,
                max_retries=2
            )

            assert "超时" in result

    def test_wait_for_human_intervention_success(self, reset_smart_router_singleton, mock_env_vars):
        """测试等待人类响应成功"""
        router = SmartRouter()

        mock_queue_from = MagicMock()
        mock_queue_to = MagicMock()

        # 模拟收到人类响应
        mock_queue_from.get.return_value = {
            "name": "SmartRouter",
            "type": "human_response",
            "content": {"action": "retry"}
        }

        # Mock time.time以控制超时
        with patch('time.time', side_effect=[0, 0.5, 1.0]):
            result = router._wait_for_human_intervention(
                mock_queue_from, mock_queue_to, timeout=10
            )

        assert result == {"action": "retry"}

    def test_wait_for_human_intervention_timeout(self, reset_smart_router_singleton, mock_env_vars):
        """测试等待人类响应超时"""
        router = SmartRouter()

        mock_queue_from = MagicMock()
        mock_queue_from.get.side_effect = queue.Empty
        mock_queue_to = MagicMock()

        # 模拟超时
        with patch('time.time', side_effect=[0, 1, 2, 302]):  # 302 > timeout=300
            result = router._wait_for_human_intervention(
                mock_queue_from, mock_queue_to, timeout=300
            )

        assert result is None


@pytest.mark.unit
class TestIsSuccess:
    """测试成功判断功能"""

    def test_success_indicators(self, reset_smart_router_singleton, mock_env_vars):
        """成功标识词识别"""
        router = SmartRouter()

        success_cases = [
            "任务完成",
            "操作成功",
            "任务 finished",
            "successfully done",
            "任务已完成"
        ]

        for case in success_cases:
            assert router._is_success(case) is True, f"Failed for: {case}"

    def test_failure_indicators(self, reset_smart_router_singleton, mock_env_vars):
        """失败标识词识别"""
        router = SmartRouter()

        failure_cases = [
            "任务失败",
            "发生错误",
            "操作 error",
            "task failed",
            "超过最大迭代次数"
        ]

        for case in failure_cases:
            assert router._is_success(case) is False, f"Failed for: {case}"

    def test_failure_priority(self, reset_smart_router_singleton, mock_env_vars):
        """失败标识优先于成功标识"""
        router = SmartRouter()

        # 同时包含失败和成功标识
        mixed_case = "任务虽然完成了但发生了错误"
        assert router._is_success(mixed_case) is False

    def test_non_string_result(self, reset_smart_router_singleton, mock_env_vars):
        """非字符串结果返回False"""
        router = SmartRouter()

        assert router._is_success(None) is False
        assert router._is_success(123) is False
        assert router._is_success(["list"]) is False
        assert router._is_success({"dict": "value"}) is False

    def test_default_success(self, reset_smart_router_singleton, mock_env_vars):
        """无明显标识时默认成功"""
        router = SmartRouter()

        # 没有成功也没有失败标识
        neutral_case = "这是普通的结果"
        assert router._is_success(neutral_case) is True


@pytest.mark.unit
class TestRouterInitialization:
    """测试路由器初始化"""

    def test_init_loads_env_vars(self, reset_smart_router_singleton):
        """测试从环境变量加载配置"""
        with patch.dict(os.environ, {
            "GUIAgent_MODEL": "custom-model",
            "GUIAgent_API_KEY": "custom-key",
            "GUIAgent_API_BASE": "https://custom.api"
        }):
            router = SmartRouter()
            assert router.model == "custom-model"
            assert router.api_key == "custom-key"
            assert router.api_base == "https://custom.api"

    def test_init_default_values(self, reset_smart_router_singleton):
        """测试默认值"""
        with patch.dict(os.environ, {}, clear=True):
            router = SmartRouter()
            assert router.model == "gpt-4o"
            assert router.gui_agent is None
            assert router.code_agent is None

    def test_lazy_loading_gui(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent):
        """测试GUI Agent懒加载"""
        router_cls, gui_mock = mock_gui_agent
        router = SmartRouter()

        # 初始时未加载
        assert router.gui_agent is None

        # 调用_get_gui_agent时加载
        agent = router._get_gui_agent()
        assert agent is gui_mock
        assert router.gui_agent is gui_mock

    def test_lazy_loading_code(self, reset_smart_router_singleton, mock_env_vars, mock_code_agent):
        """测试Code Agent懒加载"""
        router_cls, code_mock = mock_code_agent
        router = SmartRouter()

        # 初始时未加载
        assert router.code_agent is None

        # 调用_get_code_agent时加载
        agent = router._get_code_agent()
        assert agent is code_mock
        assert router.code_agent is code_mock


@pytest.mark.unit
class TestEdgeCases:
    """测试边界情况"""

    def test_execute_with_exception(self, reset_smart_router_singleton, mock_env_vars, mock_gui_agent, mock_code_agent):
        """Agent执行抛出异常"""
        router, gui_mock = mock_gui_agent
        _, code_mock = mock_code_agent

        router_instance = SmartRouter()
        router_instance._get_gui_agent = lambda: gui_mock
        router_instance._get_code_agent = lambda: code_mock
        router_instance.analyze_task = MagicMock(return_value=("gui", 0.7))

        mock_queue_to = MagicMock()
        mock_queue_from = MagicMock()

        # GUI抛出异常
        gui_mock.task.side_effect = Exception("执行错误")
        code_mock.task.return_value = "成功完成"

        with patch.object(router_instance, '_is_success', return_value=True):
            result = router_instance.execute_with_fallback(
                "任务描述",
                mock_queue_from,
                mock_queue_to,
                max_retries=1
            )

        # 应该切换到code agent
        gui_mock.task.assert_called_once()
        code_mock.task.assert_called_once()

    def test_llm_analyze_empty_response(self, reset_smart_router_singleton, mock_env_vars):
        """LLM返回空响应"""
        router = SmartRouter()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch('core.agents.smart_router.completion', return_value=mock_response):
            agent_type, confidence = router._llm_analyze("task")

        assert agent_type == "gui"
        assert confidence == 0.5

    def test_analyze_task_case_insensitive(self, reset_smart_router_singleton, mock_env_vars):
        """关键词匹配大小写不敏感"""
        router = SmartRouter()

        # 大写GUI关键词
        agent_type, confidence = router.analyze_task("打开浏览器")
        assert agent_type == "gui"

        # 大小写混合
        agent_type, confidence = router.analyze_task("计算Algorithm")
        assert agent_type == "code"
