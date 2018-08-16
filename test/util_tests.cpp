#include "catch.hpp"
#include "util.hpp"

TEST_CASE("Find next multiple", "[util]")
{
    REQUIRE(next_multiple(0, 8) == 0);
    REQUIRE(next_multiple(5, 8) == 8);
    REQUIRE(next_multiple(8, 8) == 8);
    REQUIRE(next_multiple(15, 8) == 16);
    REQUIRE(next_multiple(16, 8) == 16);
    REQUIRE(next_multiple(17, 8) == 24);
    REQUIRE(next_multiple(17, 9) == 18);
}