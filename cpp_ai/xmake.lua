add_rules("mode.debug", "mode.release")

set_languages("c++17")

target("ginger_cpp")
    set_kind("static")
    add_headerfiles("include/(**.hpp)")
    add_files("src/**.cpp")
    add_includedirs("include", {public = true})

target("ginger_tests")
    set_kind("binary")
    add_deps("ginger_cpp")
    add_files("tests/**.cpp")
    add_includedirs("include")
    
    -- Add doctest
    add_requires("doctest")
    add_packages("doctest")

    on_run(function (target)
        os.exec(target:targetfile())
    end)

target("example_aberth")
    set_kind("binary")
    add_deps("ginger_cpp")
    add_files("examples/aberth_example.cpp")
    add_includedirs("include")