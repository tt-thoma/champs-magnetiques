from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import Any, Callable
from unittest import TestCase, main

from examples import base_dir


class TestExamplesMeta(type):
    def __new__[T](
        mcs: type[T], name: str, bases: tuple[type, ...], dict: dict[str, Any]
    ) -> T:
        def new_test(function: Callable[[], None]) -> Callable[[T], None]:
            def test(self: T) -> None:
                function()

            return test

        for mod in iter_modules([base_dir]):
            if mod.name.startswith("demo"):  # mod.name.startswith("anim")
                module_path: str = f"examples.{mod.name}"
                module: ModuleType = import_module(module_path)
                dict[f"test_{mod.name.lstrip('demo_')}"] = new_test(module.main)

        return type.__new__(mcs, name, bases, dict)


class TestExamples(TestCase, metaclass=TestExamplesMeta):
    pass

    # def test_examples(self) -> None:
    #     # onerror=lambda err: warnings.warn(err, ImportWarning) (for walk_packages)
    #     for mod in iter_modules([base_dir]):
    #         if mod.name.startswith("anim"):
    #             module_path: str = f"examples.{mod.name}"
    #             with self.subTest(msg=module_path):
    #                 module: ModuleType = import_module(module_path)
    #                 module.main()


if __name__ == "__main__":
    main()
