
local TestModule = {}
TestModule.__index = TestModule

function TestModule.create(opt)
  local self = {}
  setmetatable(self, TestModule)
  
  print(opt.a or 1)
  
  self.prop1 = 10
  return self
end

local function private()
  print("private")
end

function TestModule:public()
  private()
  print(self.prop1)
end



return TestModule