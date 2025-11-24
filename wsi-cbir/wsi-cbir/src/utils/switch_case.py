from typing import Any

class SwitchCase:
    def __init__(self, var : Any):
        self.var = var
        self.cases = []
        self.matched = False
        if type(self.var) == SwitchCase._get_unmatchable: self.case = self._override
            
    def case(self, case : Any) -> bool:
        self.cases.append(case)
        if type(case)==list:
            for c in case:
                if self.var == c and not self.matched:
                    self.matched = True
                    return True
        else:
            if self.var == case and not self.matched:
                self.matched = True
                return True
        return False
    
    def _override(self, case : Any) -> bool:
        self.cases.append(case)
        return False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.matched:
            raise LookupError(f"""SwitchCase unmatched, unknown variable '{self.var}'. --OPTIONS: {self.cases}""")

    def __call__(self, case : Any) -> bool:
        self.case(case)
        
    @staticmethod
    def get_options(func):
        try:
            func(SwitchCase._get_unmatchable())
        except LookupError as e:
            msg = str(e)
            opt = msg.split('--OPTIONS: ')[-1]
            return opt
                  
    @staticmethod
    def _get_unmatchable():
        class Unmatchable:
            pass
        return Unmatchable
    
    @staticmethod
    def switch_case_function(key, options):
        with SwitchCase(key) as switch:
            for case, value in options.items():
                if switch.case(case): 
                    res = value
        return res