"""
Pytest全局配置和共享fixtures
"""
import pytest
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# 确保core模块在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

# 在导入core模块前设置全局mock
def setup_module_mocks():
    """设置模块级别的mock，避免依赖问题"""
    mock_litellm = MagicMock()
    mock_litellm.suppress_instrumentation = True
    mock_litellm.token_counter.return_value = 100
    sys.modules['litellm'] = mock_litellm

    # Mock其他依赖
    for mod_name in ['pyautogui', 'pyperclip', 'PIL', 'dotenv', 'jupyter_client',
                     'pynput', 'pynput.keyboard', 'pynput.mouse',
                     'jupyter_client.manager']:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

setup_module_mocks()


@pytest.fixture
def mock_litellm():
    """Mock litellm模块，避免真实LLM调用"""
    with patch.dict('sys.modules', {'litellm': MagicMock()}):
        mock_litellm = MagicMock()
        # 默认token计数返回固定值
        mock_litellm.token_counter.return_value = 100
        mock_litellm.suppress_instrumentation = True

        with patch.dict('sys.modules', {'litellm': mock_litellm}):
            yield mock_litellm


@pytest.fixture
def mock_env_vars():
    """设置测试环境变量"""
    env_vars = {
        "GUIAgent_MODEL": "gpt-4o",
        "GUIAgent_API_KEY": "test-gui-api-key",
        "GUIAgent_API_BASE": "https://test.gui.api",
        "CodeAgent_MODEL": "gpt-4o",
        "CodeAgent_API_KEY": "test-code-api-key",
        "CodeAgent_API_BASE": "https://test.code.api",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.fixture
def temp_dir():
    """创建临时目录用于测试文件操作"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_memory_storage(temp_dir):
    """Mock MemoryManager存储目录"""
    storage_dir = os.path.join(temp_dir, "memory_storage")
    os.makedirs(storage_dir, exist_ok=True)
    return storage_dir


@pytest.fixture
def reset_smart_router_singleton():
    """重置SmartRouter单例状态"""
    # 清除现有的单例
    from core.agents import smart_router
    smart_router._router = None
    yield
    # 测试后清理
    smart_router._router = None


@pytest.fixture
def sample_base64_image():
    """返回示例base64图片数据"""
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def mock_gui_agent():
    """Mock GUIAgent类"""
    with patch('core.agents.smart_router.GUIAgent') as mock_agent:
        instance = MagicMock()
        instance.task = MagicMock(return_value="任务完成")
        mock_agent.return_value = instance
        yield mock_agent, instance


@pytest.fixture
def mock_code_agent():
    """Mock CodeAgent类"""
    with patch('core.agents.smart_router.CodeAgent') as mock_agent:
        instance = MagicMock()
        instance.task = MagicMock(return_value="任务完成")
        mock_agent.return_value = instance
        yield mock_agent, instance


@pytest.fixture
def mock_queue():
    """创建mock队列用于测试"""
    mock_q = MagicMock()
    mock_q.put = MagicMock()
    mock_q.get = MagicMock()
    mock_q.empty = MagicMock(return_value=True)
    return mock_q


@pytest.fixture
def sample_tool_calls():
    """返回示例tool_calls数据"""
    return [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "execute_code",
                "arguments": '{"code": "print(1)"}'
            }
        }
    ]


@pytest.fixture
def sample_multiple_tool_calls():
    """返回多个tool_calls数据"""
    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "tool_a", "arguments": "{}"}
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "tool_b", "arguments": "{}"}
        }
    ]


def pytest_configure(config):
    """配置pytest自定义标记"""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "memory: mark test as a MemoryManager test")
    config.addinivalue_line("markers", "router: mark test as a SmartRouter test")
